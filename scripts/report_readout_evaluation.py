"""Audit frozen readout inference and summarize paired source/external comparisons."""
import json
from pathlib import Path
import numpy as np
from eeg_align.cache import digest

ROOT=Path('artifacts/jev_paired_qa/readout_evaluation')


def cluster_difference(labels,a,b,subjects,seed=20261018):
    people=np.unique(subjects)
    n=np.array([np.sum(subjects==s) for s in people])
    diff=np.array([np.sum((a[subjects==s]==labels[subjects==s]).astype(int)-(b[subjects==s]==labels[subjects==s]).astype(int)) for s in people])
    sampled=np.random.default_rng(seed).integers(len(people),size=(20000,len(people)))
    draws=diff[sampled].sum(1)/n[sampled].sum(1)
    return dict(accuracy_difference=float(diff.sum()/n.sum()),
        subject_bootstrap_95_percent_interval=np.quantile(draws,[.025,.975]).tolist(),
        n_subjects=len(people),resamples=len(draws),seed=seed,
        note='Exploratory paired cluster bootstrap; only nine target subjects, not a confirmatory significance claim.')


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert digest(manifest['checkpoint'])==manifest['checkpoint_sha256']
    assert json.loads((ROOT/'complete.json').read_text())['manifest_sha256']==digest(ROOT/'manifest.json')
    jobs=list(manifest['jobs'])
    jobs += [dict(name='original_fixed_head_external',dataset='external',feature='original',budget='original',folder=str(ROOT/'original_fixed_head_external')),
             dict(name='original_fixed_head_source',dataset='source',feature='original',budget='original',folder=str(ROOT.parent/'diagnostics/fixed_step100_audit'))]
    reference={};arrays={};summary={};evidence={}
    for job in jobs:
        folder=Path(job.get('folder',ROOT/job['name']))
        complete=json.loads((folder/'complete.json').read_text());r=json.loads((folder/'results.json').read_text())
        assert complete['checkpoint_sha256']==manifest['checkpoint_sha256']
        assert complete['results_sha256']==digest(folder/'results.json')
        assert complete['fixed_mi_class']
        if 'head' in job:
            assert digest(job['head'])==job['head_sha256']==complete['readout_metadata_sha256']
        loaded={}
        for case in ['matched','shuffled_within_subject']:
            a=dict(np.load(folder/f'{case}.npz',allow_pickle=False))
            assert len(a['labels'])==(1000 if job['dataset']=='source' else 2840)
            assert (a['option_permutations']==0).all()
            scores=a.get('letter_scores',a.get('letter_logits'))
            assert scores.shape==(len(a['labels']),2) and np.isfinite(scores).all()
            np.testing.assert_array_equal(scores.argmax(1),a['predictions'])
            accuracy=float(np.mean(a['predictions']==a['labels']))
            assert abs(accuracy-r[case]['overall']['accuracy'])<1e-12
            for person in np.unique(a['subjects']):
                ii=a['subjects']==person
                assert abs(float(np.mean(a['labels'][ii]==a['predictions'][ii]))-r[case]['subjects'][str(person)]['accuracy'])<1e-12
            loaded[case]=a
        matched=loaded['matched'];shuffled=loaded['shuffled_within_subject']
        for k in ['labels','subjects']:np.testing.assert_array_equal(matched[k],shuffled[k])
        for person in np.unique(matched['subjects']):
            ii=matched['subjects']==person
            np.testing.assert_array_equal(np.sort(matched['source_ids'][ii]),np.sort(shuffled['source_ids'][ii]))
        if job['dataset'] in reference:
            for case in loaded:
                for k in ['source_ids','subjects','labels']:np.testing.assert_array_equal(loaded[case][k],reference[job['dataset']][case][k])
        else:reference[job['dataset']]=loaded
        arrays[job['name']]=matched
        summary[job['name']]=dict(accuracy=r['matched']['overall']['accuracy'],shuffled_accuracy=r['shuffled_within_subject']['overall']['accuracy'],
            signal_advantage=r['matched']['overall']['accuracy']-r['shuffled_within_subject']['overall']['accuracy'],
            subjects={s:v['accuracy'] for s,v in r['matched']['subjects'].items()})
        evidence[job['name']]=dict(complete_sha256=digest(folder/'complete.json'),
            prediction_sha256={case:digest(folder/f'{case}.npz') for case in loaded})
    comparisons={}
    for budget in ['seen_classification_labels','all_training_labels']:
        joint=arrays[f'combined_{budget}_external']
        for control in [f'reve_{budget}_external','original_fixed_head_external']:
            comparisons[f'combined_{budget}_vs_{control}']=cluster_difference(joint['labels'],joint['predictions'],arrays[control]['predictions'],joint['subjects'])
    probe=ROOT.parent/'diagnostics/fixed_step100_readout'
    parity={}
    for feature in ['combined','reve']:
        for budget in ['seen_classification_labels','all_training_labels']:
            key=f'{feature}_{budget}';folder=probe/'combined' if feature=='combined' else probe
            expected=np.load(folder/f'{key}_predictions.npy')
            actual=arrays[f'{key}_source']['predictions'];np.testing.assert_array_equal(actual,expected)
            parity[key]=dict(n=len(actual),mismatches=0)
    result=dict(metrics=summary,comparisons=comparisons,source_offline_parity=parity,evidence=evidence,
        limitation='Fixed-prompt binary MI only. Repeated source validation; REVE pretraining includes 2b. No new 2b supervision or target-based parameter selection in this evaluation.')
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# Frozen readout evaluation','', '| Model | Source accuracy | 2b accuracy | 2b shuffled |','|---|---:|---:|---:|']
    for label,feature,budget in [('Original JEV head','original','fixed_head'),('Joint, 3200 labels','combined','seen_classification_labels'),('Joint, 8800 labels','combined','all_training_labels'),('REVE, 3200 labels','reve','seen_classification_labels'),('REVE, 8800 labels','reve','all_training_labels')]:
        prefix=f'{feature}_{budget}';source=summary[f'{prefix}_source'];target=summary[f'{prefix}_external']
        lines.append(f'| {label} | {source["accuracy"]:.2%} | {target["accuracy"]:.2%} | {target["shuffled_accuracy"]:.2%} |')
    lines += ['', 'All four readouts reproduce all 1,000 offline source predictions exactly.', '',
              'Joint readout restores signal-dependent classification, but does not outperform the REVE-only control. This is not evidence of a language-model benefit.', '',result['limitation']]
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'audit_complete.json').write_text(json.dumps(dict(summary_sha256=digest(ROOT/'summary.json'),report_sha256=digest(ROOT/'REPORT.md'),code_sha256=digest(__file__)),indent=2))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
