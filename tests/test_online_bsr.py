from types import SimpleNamespace
import numpy as np
import pytest
import torch

from eeg_align.online_bsr import BSRSignalStore, compatible_encoder_state


def source(samples=901, hz=200):
    x = np.random.default_rng(0).normal(size=(4, 3, samples)).astype(np.float32)
    return SimpleNamespace(eeg_data=x, sampling_rate=hz, channel_names=['C3', 'Cz', 'C4'])


def test_source_rows_full_epoch_native_channels():
    data = source()
    store = BSRSignalStore('BNCI2014004', data)
    x, names = store.batch([{'source_index': 2}, {'source_index': 0}])
    assert x.shape == (2, 3, 901) and names == ['C3', 'CZ', 'C4']
    np.testing.assert_array_equal(x.numpy(), data.eeg_data[[2, 0]])


def test_resampling_preserves_duration():
    x, _ = BSRSignalStore('BNCI2014004', source(1000, 250)).batch([{'source_index': 1}])
    assert x.shape == (1, 3, 800)


def test_unknown_montage_and_invalid_signal_fail_closed():
    data = source()
    with pytest.raises(ValueError, match='Only'):
        BSRSignalStore('SEED', data)
    data.channel_names = ['C3', 'C3', 'C4']
    with pytest.raises(ValueError, match='Duplicate'):
        BSRSignalStore('BNCI2014004', data)
    data = source()
    data.eeg_data[0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        BSRSignalStore('BNCI2014004', data).batch([{'source_index': 0}])


def test_legacy_bias_conversion_is_exact_and_nonmutating():
    p = 'encoder.layer.0.attention.attention.'
    q, v = torch.randn(8), torch.randn(8)
    original = {p+'q_bias': q, p+'v_bias': v}
    expected = {p+n for n in ('query.bias', 'key.bias', 'value.bias')}
    converted = compatible_encoder_state(original, expected)
    assert set(converted) == expected and len(original) == 2
    torch.testing.assert_close(converted[p+'query.bias'], q, rtol=0, atol=0)
    torch.testing.assert_close(converted[p+'value.bias'], v, rtol=0, atol=0)
    assert not converted[p+'key.bias'].any()


def test_gpu_gate_refuses_busy_devices(monkeypatch):
    from scripts.run_benchmark import gpu_available
    monkeypatch.setattr('subprocess.check_output', lambda *a, **k: '55518, 52\n')
    assert not gpu_available(3)
    monkeypatch.setattr('subprocess.check_output', lambda *a, **k: '1, 0\n')
    assert gpu_available(2)
