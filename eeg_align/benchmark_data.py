"""Dataset and signal adapters for task-conditioned EEG-FM-Benchmark SFT.

Bipolar signals use a minimum-norm electrode representation. This fixes each
connected component's arbitrary reference; it does not recover monopolar EEG.
No class labels are used by signal preprocessing or instruction embeddings.
"""
import json
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction

CATALOG = Path('configs/benchmark_tasks.json')


def catalog():
    return json.loads(CATALOG.read_text())['tasks']


def bipolar_matrix(channels):
    pairs = [name.split('-') for name in channels]
    if not all(len(pair) == 2 for pair in pairs):
        raise ValueError('Mixed or invalid bipolar montage')
    electrodes = list(dict.fromkeys(e for pair in pairs for e in pair))
    incidence = np.zeros((len(pairs), len(electrodes)))
    for i, (positive, negative) in enumerate(pairs):
        incidence[i, electrodes.index(positive)] = 1
        incidence[i, electrodes.index(negative)] = -1
    return np.linalg.pinv(incidence).astype(np.float32), electrodes, incidence


def to_electrodes(x, channels):
    x = np.asarray(x, dtype=np.float32)
    if any('-' in ch for ch in channels):
        _, electrodes, incidence = bipolar_matrix(channels)
        # Benchmark's EDF loader fills missing leads with zeros. They are missing
        # measurements, not constraints that the two electrodes have equal voltage.
        active = np.any(x != 0, axis=-1)
        recovered = np.zeros((len(x), len(electrodes), x.shape[-1]), dtype=np.float32)
        residual = 0.0
        for pattern in np.unique(active, axis=0):
            if not pattern.any():
                raise ValueError('All-zero EEG epoch')
            ix = np.flatnonzero(np.all(active == pattern, axis=1))
            observed = x[ix][:, pattern]
            design = incidence[pattern]
            values = np.einsum('ec,bct->bet', np.linalg.pinv(design), observed)
            reconstructed = np.einsum('ce,bet->bct', design, values)
            error = float(np.linalg.norm(reconstructed-observed) / max(np.linalg.norm(observed), 1e-12))
            residual = max(residual, error)
            if error > .05:
                raise ValueError(f'Bipolar montage inconsistent: relative residual={error}')
            recovered[ix] = values
        return recovered, electrodes, {'montage': 'minimum_norm_bipolar', 'relative_residual': residual,
                                        'missing_lead_instances': int((~active).sum()),
                                        'limitation': 'Component references unidentifiable; not recovered monopolar EEG'}
    return x, list(channels), {'montage': 'original_electrodes'}


def resample(x, source_hz):
    ratio = Fraction(200 / float(source_hz)).limit_denominator(10000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)


def segment(x, maximum=2000):
    """Preserve complete epochs using ordered <=10s chunks, never truncate a tail."""
    if x.shape[-1] < 100:
        raise ValueError('LEAF requires at least 0.5 seconds at 200 Hz')
    count = int(np.ceil(x.shape[-1] / maximum))
    # Avoid a tiny final chunk: split approximately equally instead.
    return np.array_split(x, count, axis=-1)



def classification_metrics(labels, probabilities, classes):
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(probabilities)[:, :classes].argmax(-1)
    cm = np.zeros((classes, classes), dtype=int)
    np.add.at(cm, (y, pred), 1)
    support = cm.sum(1)
    recall = np.divide(np.diag(cm), support, out=np.full(classes, np.nan), where=support > 0)
    return dict(n=len(y), accuracy=float(np.mean(y == pred)),
                balanced_accuracy=float(np.nanmean(recall)),
                supported_classes=int((support>0).sum()), confusion_matrix=cm.tolist())


def load_pickle(path):
    """Read trusted local EEGData pickles without LEAF's top-level utils collision."""
    import pickle
    import importlib.util
    spec = importlib.util.spec_from_file_location('benchmark_eeg_schema', Path(__file__).resolve().parents[2]/'EEG-FM-Benchmark/utils/EEGDataLoader.py')
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    class LocalUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'utils.EEGDataLoader' and name == 'EEGData':
                return schema.EEGData
            return super().find_class(module, name)
    with Path(path).open('rb') as handle:
        return LocalUnpickler(handle).load()

