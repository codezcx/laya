"""Complete-pair summaries, explicit exposure strata, and automatic overnight report."""
import csv
import json
from .common import CONFIG, OUT, write_json

LABELS=dict(last_only='Second linear',first_only='First linear',both='Both linears',tower_both='Tower LoRA + both')


def aggregate(points,plan):
    cases={c['key']:c for c in plan['cases']}; episodes={e['key']:e for e in plan['episodes']}
    index={}
    for p in points:
        ep=p['episode']; case=p['case']; n=ep['support_total']; key=(case['key'],ep['arm'],n,p['step'])
        if (ep['key'] not in episodes or ep!=episodes[ep['key']] or case['key']!=ep['case_key'] or
                p['step'] not in CONFIG['eval_steps'] or p['arm']!=ep['arm'] or p['support_n']!=n or
                case['support_ids']!=plan['support_sets'][case['key']][str(n)] or
                case['query_ids']!=cases[case['key']]['query_ids']):
            raise RuntimeError('Unexpected curve identity')
        if key in index: raise RuntimeError('Duplicate curve point')
        index[key]=p
    def collect(selected):
        by_size={}
        if not selected: return by_size
        for n in CONFIG['support_sizes']:
            arms=CONFIG['arms'] if n==100 else CONFIG['dose_arms']; ref='last_only' if n==100 else 'both'
            values={}
            for step in CONFIG['eval_steps']:
                if any((c['key'],a,n,step) not in index for c in selected for a in arms): continue
                entries={}
                for c in selected:
                    baseline=index[c['key'],ref,n,step]['baseline']
                    if any(index[c['key'],a,n,step]['baseline']!=baseline for a in arms):
                        raise RuntimeError('Baseline differs across paired conditions')
                for arm in arms:
                    group=[index[c['key'],arm,n,step] for c in selected]
                    metrics={f'{part}_{metric}':sum(p[part][metric] for p in group)/len(group)
                             for part in ('query','support') for metric in ('accuracy','balanced_accuracy','ce')}
                    diffs=[p['query']['balanced_accuracy']-index[p['case']['key'],ref,n,step]['query']['balanced_accuracy'] for p in group]
                    metrics.update(baseline_bacc=sum(p['baseline']['balanced_accuracy'] for p in group)/len(group),
                        delta_vs_reference_pp=100*sum(diffs)/len(diffs),
                        delta_vs_unadapted_pp=100*sum(p['query']['balanced_accuracy']-p['baseline']['balanced_accuracy'] for p in group)/len(group),
                        negative_vs_reference_users=sum(d < -1e-10 for d in diffs),
                        negative_vs_unadapted_users=sum(p['query']['balanced_accuracy']<p['baseline']['balanced_accuracy']-1e-10 for p in group))
                    entries[arm]=dict(users=len(group),case_ids=[c['key'] for c in selected],reference_arm=ref,**metrics)
                values[str(step)]=entries
            by_size[str(n)]=values
        return by_size
    matched={}; exposure={}; unseen={}
    for name in CONFIG['datasets']:
        selected=[c for c in cases.values() if c['dataset']==name]
        matched[name]=collect(selected)
        exposure[name]={s:collect([c for c in selected if c['exposure_stratum']==s])
                        for s in sorted({c['exposure_stratum'] for c in selected})}
        subset=[c for c in selected if not c['shared_train_seen_subject']]
        if subset: unseen[name]=collect(subset)
    def equal_task(groups):
        output={}
        for n in CONFIG['support_sizes']:
            values={}
            for step in CONFIG['eval_steps']:
                if not groups or not all(str(step) in g[str(n)] for g in groups.values()): continue
                arms=CONFIG['arms'] if n==100 else CONFIG['dose_arms']; values[str(step)]={}
                for arm in arms:
                    rows=[g[str(n)][str(step)][arm] for g in groups.values()]
                    fields=('query_accuracy','query_balanced_accuracy','query_ce','support_accuracy','support_ce',
                            'baseline_bacc','delta_vs_reference_pp','delta_vs_unadapted_pp')
                    values[str(step)][arm]={field:sum(r[field] for r in rows)/len(rows) for field in fields}
            output[str(n)]=values
        return output
    return dict(matched=matched,exposure=exposure,equal_task_all_exposures=equal_task(matched),
        equal_task_unseen_subjects=equal_task(unseen),unseen_datasets=list(unseen),
        curve_points=len(index),expected_curve_points=len(episodes)*len(CONFIG['eval_steps']),
        expected_unique_results=len(episodes),query_used_for_selection=False,independent_test_estimate=False)


def plot(report,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    directory.mkdir(parents=True,exist_ok=True)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']; steps=CONFIG['eval_steps']
    for filename,metrics in [('location_curves',[('query_balanced_accuracy','Query BAcc (%)',100)]),
                             ('support_query_curves',[('support_accuracy','Support Acc (%)',100),('support_ce','Support CE',1),('query_ce','Query CE',1)])]:
        fig,axes=plt.subplots(len(metrics),4,figsize=(16,3.4*len(metrics)),squeeze=False,constrained_layout=True)
        for col,name in enumerate(CONFIG['datasets']):
            values=report['matched'][name]['100']
            for row,(metric,label,scale) in enumerate(metrics):
                ax=axes[row,col]
                for arm,color,marker in zip(CONFIG['arms'],colors,['o','s','^','D']):
                    ax.plot(steps,[scale*values[str(s)][arm][metric] if str(s) in values else np.nan for s in steps],
                            label=LABELS[arm],color=color,marker=marker,markersize=3)
                if not values: ax.text(.5,.5,'Waiting for all paired users / arms',ha='center',transform=ax.transAxes,fontsize=8)
                ax.set(title=name,xlabel='Personal optimizer step',ylabel=label); ax.grid(alpha=.25)
                ax.spines[['top','right']].set_visible(False)
        axes[0,0].legend(fontsize=7)
        fig.suptitle('100 support samples; fixed cohorts; exploratory query; shared-training exposure is mixed')
        for ext in ('png','pdf'): fig.savefig(directory/f'{filename}.{ext}',dpi=160)
        plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,4),constrained_layout=True); sizes=sorted(CONFIG['support_sizes'])
    for ax,name in zip(axes,CONFIG['datasets']):
        for arm,color,marker in zip(CONFIG['dose_arms'],colors[2:],['^','D']):
            y=[report['matched'][name][str(n)].get('100',{}).get(arm,{}).get('query_balanced_accuracy',np.nan)*100 for n in sizes]
            ax.plot(sizes,y,color=color,marker=marker,label=LABELS[arm])
        ax.set(title=name,xlabel='Nested support samples',ylabel='Step100 query BAcc (%)'); ax.set_xticks(sizes); ax.grid(alpha=.25)
    axes[0].legend(fontsize=8); fig.suptitle('Fixed100updates; query membership unchanged; SEED/HMC counts are windows/epochs')
    for ext in ('png','pdf'): fig.savefig(directory/f'support_size_curves.{ext}',dpi=160)
    plt.close(fig)


def write_report(report,plan):
    lines=['# Personal LoRA location and support-size study','',
        f"Completed: {report['completed_unique_results']}/{report['expected_unique_results']} episodes; {report['curve_points']}/{report['expected_curve_points']} evaluation points.",'',
        'Exploratory personal queries, not a blind benchmark. Fixed step100; no query-based checkpoint selection. Empty comparisons wait for every paired user/arm.','',
        '## Source dataset membership (historical parent cache)','',
        '| Dataset | Processed windows total (train/valid/test) | Task (classes) | Personal split protocol | Subjects total (train/valid/test) |',
        '| --- | --- | --- | --- | --- |']
    task_types={'bcic_2a':'MI (4)','seed':'Emotion (3)','hmc':'Sleep staging (5)','Nakanishi2015':'SSVEP (12)'}
    for name in CONFIG['datasets']:
        v=plan['inventory'][name]
        def counts(key): return '/'.join(str(v[key].get(s,0)) for s in ('train','valid','test'))
        lines.append(f"| {name} | {v['windows_total']} ({counts('window_counts')}) | {task_types[name]} | {CONFIG['cohorts'][name]['policy']} | {v['subjects_total']} ({counts('subject_counts')}) |")
    lines+=['','## Fixed step100 paired results','']
    groups={**{name:report['matched'][name] for name in CONFIG['datasets']},
            'Equal-task mean, all exposure strata':report['equal_task_all_exposures'],
            'Equal-task mean, shared-training-unseen subjects only':report['equal_task_unseen_subjects']}
    for name,group in groups.items():
        lines += [f'### {name}','','| Support | Method | Query BAcc (%) | Query Acc (%) | Support Acc (%) | Gain over unadapted (pp) |',
                  '| ---: | --- | ---: | ---: | ---: | ---: |']
        wrote=False
        for n in CONFIG['support_sizes']:
            for arm,values in group[str(n)].get('100',{}).items():
                lines.append(f"| {n} | {LABELS[arm]} | {100*values['query_balanced_accuracy']:.2f} | {100*values['query_accuracy']:.2f} | {100*values['support_accuracy']:.2f} | {values['delta_vs_unadapted_pp']:+.2f} |")
                wrote=True
        if not wrote: lines+=['','Pending complete paired results.']
        lines+=['']
    lines += ['## Interpretation boundaries','',
        '- All 21 people are separate resets, not pooled personal training. Shared-training exposure strata are in summary.json and curve_points.csv.',
        '- 2a A01-A05 and Nakanishi queries are in the historical shared-training pool; SEED users are seen but query source trials are excluded from shared train.',
        '- A06/A07 and HMC s7/s9 reuse historically authorized original-valid users. This is not a blind holdout.',
        '- 24/48/100 are nested samples, not equal epochs: 100 updates ×16 / support_n gives 66.67/33.33/16 equivalent epochs.',
        '- First-only/second-only are tested at100 only; no claim about their24/48 behavior. No claim that100 is globally optimal.',
        '- Do not use the highest observed query point as a new primary score. Use fixed100 and report full curves.',
        '', '[Location curves](figures/location_curves.png) · [Support/query curves](figures/support_query_curves.png) · [Support-size curves](figures/support_size_curves.png)','']
    path=OUT/'report.md'; tmp=path.with_suffix('.md.tmp'); tmp.write_text('\n'.join(lines)); tmp.replace(path)


_last_curve_count=-1


def summary():
    global _last_curve_count
    plan=json.loads((OUT/'plan.json').read_text())
    paths=sorted((OUT/'jobs').glob('*/cases/*/*/curves/step*.json'))
    points=[json.loads(p.read_text()) for p in paths]
    if any(p['stamp']['source_hashes']!=plan['source_hashes'] for p in points): raise RuntimeError('Curve provenance changed')
    report=aggregate(points,plan)
    report['completed_unique_results']=len(list((OUT/'jobs').glob('*/cases/*/*/result.json')))
    write_json(OUT/'summary.json',report); write_report(report,plan)
    if len(points)!=_last_curve_count:
        with (OUT/'curve_points.csv').open('w',newline='') as handle:
            writer=csv.writer(handle)
            writer.writerow(['dataset','user','exposure','arm','support_n','step','query_bacc','query_acc','query_ce','support_acc','support_ce','query_n','equivalent_epochs'])
            for p in points:
                writer.writerow([p['case']['dataset'],p['case']['subject'],p['case']['exposure_stratum'],p['arm'],p['support_n'],p['step'],
                    p['query']['balanced_accuracy'],p['query']['accuracy'],p['query']['ce'],p['support']['accuracy'],p['support']['ce'],len(p['case']['query_ids']),p['equivalent_epochs']])
        plot(report,OUT/'figures'); _last_curve_count=len(points)
    return report
