"""Audit all recorded probes and derive publication numbers without rerunning models."""
import csv,json,math
from pathlib import Path
from statistics import mean
from data import KNOWN_TARGETS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def main():
    primary=json.loads((OUT/'results.json').read_text())
    matched=json.loads((OUT/'matched_control.json').read_text())
    memory=json.loads((OUT/'memory/results.json').read_text())
    dev=json.loads((OUT/'development.json').read_text())
    reload=json.loads((OUT/'reload_check.json').read_text())
    assert primary['base_unchanged'] and matched['base_unchanged']
    assert reload['fresh_process_reload_matches']
    rows=[]
    def audit(ev,stage,condition,seed,step,kind):
        assert len(ev['items'])==ev['n']
        assert sum(x['agreement'] for x in ev['items'])==ev['agreement']
        if ev['correct'] is not None:assert sum(x['correct'] for x in ev['items'])==ev['correct']
        for i,x in enumerate(ev['items']):
            assert (x['prediction']==x['base_prediction'])==x['agreement']
            if 'correct' in x:
                assert (x['prediction']==x['target'])==x['correct']
                assert 0<=x['target_probability']<=1
            rows.append(dict(stage=stage,condition=condition,seed=seed,step=step,probe=kind,
                             prompt=x['prompt'],prediction=x['prediction'],base_prediction=x['base_prediction'],
                             target=x.get('target',''),correct=x.get('correct',''),agreement=x['agreement']))
    for key,ev in primary['baseline'].items():audit(ev,'test','frozen_'+key,'',0,key)
    for stage,runs in [('dev',dev),('test',primary['runs']),('matched',matched['runs'])]:
        for r in runs:
            assert all(math.isfinite(x['gradient_norm']) and x['gradient_norm']>0 for x in r['curve'])
            for cp in r['checkpoints']:
                for kind in ['facts','retention']:audit(cp[kind],stage,r['method'],r['seed'],cp['step'],kind)
    for method,selected in primary['selection'].items():
        candidates=[]
        for r in dev:
            if r['method']!=method:continue
            for c in r['checkpoints']:
                score=c['facts']['correct']/c['facts']['n']-2*(1-c['retention']['agreement']/c['retention']['n'])
                candidates.append((score,-c['step'],-r['lr'],r['kl_weight']))
        assert max(candidates)==(selected['selection_score'],-selected['steps'],-selected['lr'],selected['kl_weight'])
    with (OUT/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def summarize(rs):
        a=[r['checkpoints'][-1]['facts']['correct'] for r in rs]
        b=[r['checkpoints'][-1]['retention']['agreement'] for r in rs]
        k=[sum(x['prediction'].strip()==target for x,target in zip(r['checkpoints'][-1]['retention']['items'][:8],KNOWN_TARGETS)) for r in rs]
        return dict(fact_correct=a,fact_mean=mean(a),fact_total=72,fact_percent=mean(a)/72*100,
                    retention_agreement=b,retention_mean=mean(b),retention_total=48,retention_percent=mean(b)/48*100,
                    known_answer_correct=k,known_answer_total=8)
    summary=dict(audit='passed',audited_probe_records=len(rows),
                 selected={m:summarize([r for r in primary['runs'] if r['method']==m]) for m in primary['protocol']['methods']},
                 matched_unprotected=summarize(matched['runs']),
                 baseline_fact_correct=primary['baseline']['facts']['correct'],
                 oracle_fact_correct=primary['baseline']['relevant_fact']['correct'],
                 baseline_known_answers=sum(x['prediction'].strip()==t for x,t in zip(primary['baseline']['retention']['items'][:8],KNOWN_TARGETS)),
                 reload_check=reload,
                 synthetic_memory={})
    for condition in memory['runs'][0]['conditions']:
        values=[]
        for r in memory['runs']:
            c=r['conditions'][condition]
            assert sum(c['episode_correct'])==c['total_correct']
            assert math.isclose(c['total_correct']/c['total_predictions'],c['accuracy'],abs_tol=1e-7)
            assert r['gradient_write_check'] and r['slow_encoder_unchanged_during_test']
            values.append(c['total_correct']/c['total_predictions']*100)
        summary['synthetic_memory'][condition]=dict(percent_by_seed=values,mean_percent=mean(values))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
