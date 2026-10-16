"""Independent linear/nonlinear readouts of frozen representations, no JEV."""
import argparse
import json
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn

from .data import DATASETS, digest, prepare_dataset, save_json

LINEAR_C = (.001, .01, .1, 1.)
SEEDS = (42, 43)
MAX_EPOCHS = 120


def labels_for(rows, target):
    if target == 'task':
        return np.array([r['label'] for r in rows]), ['left_hand', 'right_hand']
    names = sorted({r['subject'] for r in rows if r['split'] == 'validation'}, key=int)
    mapping = {name: i for i, name in enumerate(names)}
    # Source training people must never enter the identity-probe dataset.
    return np.array([mapping.get(r['subject'], -1) for r in rows]), names


def sample_weights(rows, ids):
    keys = [(rows[i]['subject'], rows[i]['label']) for i in ids]
    counts = {k: keys.count(k) for k in set(keys)}
    weights = np.array([1. / counts[k] for k in keys], dtype=np.float32)
    return weights / weights.mean()


def scores(rows, ids, y, pred, target, with_ci=False):
    people = sorted({rows[i]['subject'] for i in ids}, key=int)
    subjects = np.array([rows[i]['subject'] for i in ids])
    by_subject = {}
    for person in people:
        mask = subjects == person
        if target == 'task':
            value = balanced_accuracy_score(y[mask], pred[mask])
        else:
            value = np.mean(y[mask] == pred[mask])
        by_subject[person] = float(value)
    result = dict(accuracy=float(np.mean(y == pred)),
                  balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                  subject_macro=float(np.mean(list(by_subject.values()))),
                  per_subject=by_subject, n=len(y), correct=int(np.sum(y == pred)))
    result['selection_score'] = result['subject_macro'] if target == 'task' else result['balanced_accuracy']
    if with_ci:
        values = np.array(list(by_subject.values()))
        bootstrap = np.random.default_rng(20261016).choice(values, (2000, len(values)), replace=True).mean(1)
        result['subject_bootstrap_95ci'] = np.quantile(bootstrap, [.025, .975]).tolist()
        result['ci_note'] = 'Descriptive resampling of evaluated people; identity class set/model kept fixed.'
    return result


def fit_linear(x_train, y_train, x_val, y_val, weights, val_score, folder):
    best, best_score, history = None, -np.inf, []
    for c in LINEAR_C:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(
            C=c, solver='lbfgs', max_iter=1000, tol=1e-5, random_state=42))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            pipe.fit(x_train, y_train, logisticregression__sample_weight=weights)
        value = val_score(pipe.predict(x_val))
        item = dict(C=c, validation_score=value,
                    iterations=pipe[-1].n_iter_.tolist(),
                    convergence_warning=any(issubclass(w.category, ConvergenceWarning) for w in caught))
        history.append(item)
        print(json.dumps(dict(probe='linear', **item)), flush=True)
        if value > best_score:
            best, best_score = pipe, value
    joblib.dump(best, folder / 'model.joblib')
    save_json(folder / 'selection.json', dict(candidates=history, selected_C=float(best[-1].C),
                                              validation_score=best_score, refit=False))
    return best.predict_proba, best.predict, dict(selected_C=float(best[-1].C), validation_score=best_score)


def make_mlp(input_dim, classes):
    return nn.Sequential(nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, classes))


def fit_mlp(x_train, y_train, x_val, y_val, weights, val_score, seed, folder,
            max_epochs=MAX_EPOCHS, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(x_train)
    xt = torch.from_numpy(scaler.transform(x_train).astype(np.float32)).to(device)
    xv = torch.from_numpy(scaler.transform(x_val).astype(np.float32)).to(device)
    yt = torch.from_numpy(y_train).long().to(device)
    wt = torch.from_numpy(weights).to(device)
    classes = len(np.unique(y_train))
    if not np.array_equal(np.unique(y_train), np.arange(classes)):
        raise ValueError('Missing identity class in train partition')
    model = make_mlp(x_train.shape[1], classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed)
    best, best_score, best_epoch = None, -np.inf, 0
    with (folder / 'epochs.jsonl').open('x', buffering=1) as log:
        for epoch in range(1, max_epochs + 1):
            model.train()
            order = torch.randperm(len(xt), generator=generator).to(device)
            total_loss = 0.
            for start in range(0, len(order), 256):
                idx = order[start:start + 256]
                optimizer.zero_grad(set_to_none=True)
                loss = (nn.functional.cross_entropy(model(xt[idx]), yt[idx], reduction='none') * wt[idx]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total_loss += loss.item() * len(idx)
            if epoch == 1 or epoch % 5 == 0 or epoch == max_epochs:
                model.eval()
                with torch.no_grad():
                    pred = model(xv).argmax(1).cpu().numpy()
                value = val_score(pred)
                item = dict(epoch=epoch, loss=total_loss / len(xt), validation_score=value)
                log.write(json.dumps(item) + '\n')
                if value > best_score:
                    best_score, best_epoch = value, epoch
                    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if epoch % 20 == 0:
                    print(json.dumps(dict(probe='mlp', seed=seed, **item)), flush=True)
                if epoch - best_epoch >= 25:
                    break
    model.load_state_dict(best)
    model.eval()
    config = dict(seed=seed, input_dim=x_train.shape[1], classes=classes, hidden=128,
                  lr=.001, weight_decay=.01, dropout=.1, max_epochs=max_epochs,
                  best_epoch=best_epoch, stopped_epoch=epoch, validation_score=best_score,
                  batch_size=256, early_stop_patience=25, refit=False)
    torch.save(dict(state_dict=best, config=config), folder / 'model.pt')
    joblib.dump(scaler, folder / 'scaler.joblib')
    save_json(folder / 'selection.json', config)
    def probabilities(x):
        values = []
        with torch.no_grad():
            for start in range(0, len(x), 512):
                tensor = torch.from_numpy(scaler.transform(x[start:start + 512]).astype(np.float32)).to(device)
                values.append(model(tensor).softmax(1).cpu().numpy())
        return np.concatenate(values)
    return probabilities, lambda x: probabilities(x).argmax(1), config


def check_code(root):
    manifest = json.loads((root / 'source_snapshot.json').read_text())
    for path, expected in manifest['hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Source changed after launch: {path}')


def run_dataset(dataset, root, device='cpu'):
    root = Path(root)
    folder = root / dataset
    folder.mkdir(exist_ok=False)
    try:
        check_code(root)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        started = time.monotonic()
        with threadpool_limits(limits=4):
            rows, protocols, features = prepare_dataset(dataset, folder)
            check_code(root)
            completed = 0
            with (folder / 'results.jsonl').open('x', buffering=1) as log:
                for protocol in protocols:
                    target = protocol['target']
                    y, names = labels_for(rows, target)
                    train, val, test = [np.asarray(protocol[key], dtype=int) for key in ('train', 'validation', 'test')]
                    if any(np.min(y[part]) < 0 for part in (train, val, test)):
                        raise ValueError('Non-panel identity found')
                    weights = sample_weights(rows, train)
                    def val_score(pred):
                        return scores(rows, val, y[val], pred, target)['selection_score']
                    for feature, x in features.items():
                        for probe, seed in [('linear', 42), ('mlp', 42), ('mlp', 43)]:
                            check_code(root)
                            name = f'{protocol["name"]}__{feature}__{probe}_seed{seed}'
                            run = folder / name
                            run.mkdir()
                            save_json(folder / 'status.json', dict(phase='training', active=name, completed=completed,
                                                                 total=len(protocols) * len(features) * 3))
                            print(f'TRAIN {dataset} {name}', flush=True)
                            if probe == 'linear':
                                prob_fn, pred_fn, selection = fit_linear(x[train], y[train], x[val], y[val],
                                                                        weights, val_score, run)
                            else:
                                prob_fn, pred_fn, selection = fit_mlp(x[train], y[train], x[val], y[val],
                                                                     weights, val_score, seed, run, device=device)
                            # Test partition is first read by the selected readout only here.
                            probs = prob_fn(x[test])
                            pred = probs.argmax(1)
                            test_scores = scores(rows, test, y[test], pred, target, with_ci=True)
                            majority = np.bincount(y[train]).argmax()
                            dummy = scores(rows, test, y[test], np.full(len(test), majority), target)
                            np.savez_compressed(run / 'test_predictions.npz', indices=test, source_indices=np.array(
                                [rows[i]['source_index'] for i in test]), truth=y[test], probabilities=probs)
                            result = dict(dataset=dataset, protocol=protocol['name'], target=target,
                                domain=protocol['domain'], feature=feature, probe=probe, seed=seed,
                                class_names=names, chance=1. / len(names), selection=selection,
                                n_train=len(train), n_validation=len(val), n_test=len(test),
                                train=scores(rows, train, y[train], pred_fn(x[train]), target),
                                validation=scores(rows, val, y[val], pred_fn(x[val]), target),
                                test=test_scores, dummy_test=dummy,
                                confusion_matrix=confusion_matrix(y[test], pred, labels=np.arange(len(names))).tolist(),
                                run=name, predictions_sha256=digest(run / 'test_predictions.npz'))
                            save_json(run / 'result.json', result)
                            log.write(json.dumps(result, allow_nan=False) + '\n')
                            completed += 1
                            print(json.dumps(dict(done=name, test=test_scores['selection_score'], completed=completed)), flush=True)
            save_json(folder / 'status.json', dict(phase='complete', completed=completed,
                                                   elapsed_seconds=time.monotonic() - started))
    except BaseException as exc:
        save_json(folder / 'status.json', dict(phase='failed', error=repr(exc)))
        traceback.print_exc()
        raise


def summarize(root):
    root = Path(root)
    results = []
    statuses = {}
    for ds in DATASETS:
        path = root / ds / 'results.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    # Another worker may still be appending its current result.
                    continue
        path = root / ds / 'status.json'
        statuses[ds] = json.loads(path.read_text()) if path.exists() else {'phase': 'pending'}
    save_json(root / 'summary.json', dict(status=statuses, results=results))
    lines = ['# FBCSP / REVE 特征探测结果', '',
             '自动汇总；只有已完成并保存测试预测的探针才列出成绩。任务是左右手二分类，不是 MI/情绪任务路由。', '',
             '线性探针为逻辑回归；MLP 两个 seed 分开记录。主指标：任务为被试平均平衡准确率，身份为平衡准确率。', '',
             'Cho 身份成绩只是同记录索引块隔离诊断，不是跨时间/跨会话证据。Lee 跨会话，Dreyer 跨 run。', '',
             '身份面板仅使用未参与 CSP 拟合的源留出被试；不使用 2a/2b。REVE 预训练数据重叠未排除。', '',
             '| 数据集 | 测试 | 特征 | 探针 | seed | 测试 N | 随机基准 | 主指标 | 描述性 95% CI |',
             '|---|---|---|---|---:|---:|---:|---:|---|']
    for r in results:
        lo, hi = r['test']['subject_bootstrap_95ci']
        lines.append(f'| {r["dataset"]} | {r["protocol"]} | {r["feature"]} | {r["probe"]} | {r["seed"]} | '
                     f'{r["n_test"]} | {r["chance"]:.2%} | {r["test"]["selection_score"]:.2%} | [{lo:.2%}, {hi:.2%}] |')
    lines += ['', '## 状态', ''] + [f'- {ds}: {state["phase"]}' for ds, state in statuses.items()]
    lines += ['', '## 解释边界', '',
              '- 准确率是可解码性证据，不是信息量占比；低成绩不证明统计独立。',
              '- 身份信息可能包括稳定采集差异；高身份准确率不能单独证明损害任务泛化。',
              '- 两个 seed 不是两个独立人群。CI 重采样测试被试，身份分类器及类别集合保持固定，只作描述性参考。',
              '- 源留出被试在此前实验已被使用；这是探索性诊断，不是新的盲测。', '']
    (root / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        summarize(args.root)
    elif args.dataset:
        run_dataset(args.dataset, args.root, args.device)
    else:
        parser.error('Choose --dataset or --summarize')


if __name__ == '__main__':
    main()
