"""Export/follow authoritative JSON metrics without allocating a training model.

Accuracies use fractions (0..1). Validation loss is cross entropy reconstructed
from saved semantic-class probabilities, with no regularization term.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def training_scalars(row):
    result = {'train/overall/loss': row['loss'],
              'train/overall/accuracy': row['train_accuracy']}
    for dataset, values in row['train_by_dataset'].items():
        for key in ['loss','accuracy']:
            result[f'train/datasets/{dataset}/{key}'] = values[key]
    for name, value in row.get('learning_rates', {}).items():
        result[f'learning_rate/{name}'] = value
    for key in ['gradient_norm','encoder_gradient_norm']:
        if key in row: result[f'diagnostics/{key}'] = row[key]
    return result


def validation_scalars(report, predictions):
    result = {}
    for order, probability_key, prefix in [
        ('normal','probabilities','validation'),
        ('reversed','alternate_probabilities','validation_reversed'),
    ]:
        if order not in report: continue
        scores = []
        for dataset, entry in report[order].items():
            metric = entry['epoch']
            samples = [r for r in predictions if r['dataset'] == dataset]
            assert len(samples) == metric['n'] > 0
            losses = [-math.log(max(float(r[probability_key][int(r['label'])]),1e-30)) for r in samples]
            loss = sum(losses)/len(losses)
            for key in ['accuracy','balanced_accuracy']:
                result[f'{prefix}/datasets/{dataset}/{key}'] = metric[key]
            result[f'{prefix}/datasets/{dataset}/loss'] = loss
            scores.append((metric['n'],metric['accuracy'],metric['balanced_accuracy'],loss))
        n = sum(v[0] for v in scores)
        for column, name in [(1,'accuracy'),(2,'balanced_accuracy'),(3,'loss')]:
            result[f'{prefix}/overall/{name}_macro'] = sum(v[column] for v in scores)/len(scores)
            if name != 'balanced_accuracy':
                result[f'{prefix}/overall/{name}_pooled'] = sum(v[0]*v[column] for v in scores)/n
    for dataset, value in report.get('consistency', {}).items():
        result[f'validation/datasets/{dataset}/option_consistency'] = value
    return result


def emit(writer, values, step):
    for name, value in values.items():
        assert math.isfinite(float(value)), (name,value)
        writer.add_scalar(name,float(value),int(step))


def jsonl(path):
    if not path.exists(): return []
    lines = path.read_text().splitlines()
    result = []
    for i,line in enumerate(lines):
        try: result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1: raise
    return result


def export_run(root, out, branches, follow=False, interval=10):
    from torch.utils.tensorboard import SummaryWriter
    out.mkdir(parents=True, exist_ok=True)
    state_path = out/'export_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        source=str(root.resolve()), exported={}, definitions={
            'accuracy_units':'fraction 0..1',
            'train':'sampled minibatch means per optimizer update; datasets have equal n in this run',
            'validation_macro':'equal weight per dataset; accuracy and balanced accuracy are separate',
            'validation_pooled':'weight by validation example count',
            'validation_loss':'mean -log p(true semantic class), reconstructed from prediction JSON',
            'independence':'validation checkpoints reuse fixed rows; no test data'})
    assert state['source'] == str(root.resolve())
    writers = {name:SummaryWriter(str(out/name),flush_secs=10) for name in branches}
    try:
        while True:
            added = 0
            for name in branches:
                folder = root/name
                for row in jsonl(folder/'metrics.jsonl'):
                    kind = 'train' if 'loss' in row else 'validation' if 'validation' in row else None
                    if kind is None: continue
                    key = f"{name}/{kind}/{row['step']}"
                    fingerprint = hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
                    if key in state['exported']:
                        assert state['exported'][key] == fingerprint, f'Existing metric changed: {key}'
                        continue
                    if kind == 'train':
                        values = training_scalars(row)
                    else:
                        path = folder/f"validation_{row['step']:04d}.json"
                        if not path.exists(): continue
                        saved = json.loads(path.read_text())
                        assert saved['metrics'] == row['validation']
                        values = validation_scalars(row['validation'],saved['predictions'])
                    emit(writers[name],values,row['step'])
                    state['exported'][key] = fingerprint
                    added += 1
            for writer in writers.values(): writer.flush()
            tmp = state_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
            if added: print(json.dumps(dict(added=added,total=len(state['exported']),out=str(out))),flush=True)
            if not follow or all((root/name/'complete.json').exists() or (root/name/'stopped.json').exists()
                                 for name in branches): break
            pipeline = root/'pipeline.json'
            if pipeline.exists() and json.loads(pipeline.read_text()).get('status') == 'failed': break
            time.sleep(interval)
    finally:
        for writer in writers.values(): writer.close()


def export_probe(folder, out, datasets=None):
    from torch.utils.tensorboard import SummaryWriter
    assert (folder/'complete.json').exists()
    out.mkdir(parents=True,exist_ok=False)
    metrics = json.loads((folder/'metrics.json').read_text())
    if datasets: metrics = {name:metrics[name] for name in datasets}
    predictions = jsonl(folder/'predictions.jsonl')
    all_values = {}
    with SummaryWriter(str(out)) as writer:
        writer.add_text('method','Frozen encoder + fitted linear probe; final full-split metrics, '
                        'not an optimizer-step training curve. Cross entropy excludes regularization.',0)
        for split in ['train','validation']:
            report = dict(normal={ds:dict(epoch=m[split]) for ds,m in metrics.items()})
            rows = [r for r in predictions if r['split'] == split and r['dataset'] in metrics]
            values = validation_scalars(report,rows)
            values = {k.replace('validation/',f'probe/{split}/',1):v for k,v in values.items()}
            emit(writer,values,0);all_values.update(values)
    (out/'export.json').write_text(json.dumps(dict(source=str(folder.resolve()),
        datasets=list(metrics),step=0,scalars=all_values),indent=2)+'\n')
    print(json.dumps(dict(out=str(out),scalars=len(all_values))),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',type=Path)
    group.add_argument('--probe',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--branches',nargs='+',default=['updating','frozen'])
    parser.add_argument('--datasets',nargs='+')
    parser.add_argument('--follow',action='store_true')
    parser.add_argument('--interval',type=float,default=10)
    args = parser.parse_args()
    if args.probe:
        if args.follow: parser.error('--follow is for training JSONL logs')
        export_probe(args.probe,args.out,args.datasets)
    else:
        if args.datasets: parser.error('--datasets is for probe exports')
        assert args.interval>0
        export_run(args.run,args.out,args.branches,args.follow,args.interval)


if __name__ == '__main__': main()
