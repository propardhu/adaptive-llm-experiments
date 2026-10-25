"""Audit stored predictions without PyTorch or a model download."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='results');args=ap.parse_args()
    root=Path(args.results);r=json.loads((root/'results.json').read_text())
    total=0;aggregates={};targeted={};hashes={}
    for run in r['runs']:
        path=root/f"predictions-{run['seed']}.json"
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        p=json.loads(path.read_text());truth=p['target']
        for name, predictions in p['conditions'].items():
            assert len(predictions)==len(truth)
            per=[]
            for ys,ps in zip(truth,predictions):
                assert len(ys)==len(ps)==r['protocol']['test_facts']
                assert all(0<=v<r['protocol']['value_classes'] for v in ps)
                per.append(sum(a==b for a,b in zip(ys,ps)))
            result=run['conditions'][name]
            count=sum(per);den=sum(map(len,truth));total+=den
            assert per==result['episode_correct']
            assert count==result['correct'] and den==result['total']
            assert abs(count/den-result['accuracy'])<1e-6
            aggregates.setdefault(name,[0,0]);aggregates[name][0]+=count;aggregates[name][1]+=den
        assert p['conditions']['clean_memory']==p['conditions']['reset_and_replay']==p['conditions']['source_filtered']
        for ys,indices,wrong in zip(truth,p['attacked_indices'],p['false_values']):
            assert len(set(indices))==len(indices)==len(ys)//4
            assert all(ys[i]!=w for i,w in zip(indices,wrong))
        for name,count in [('exact_lookup_clean',32),('exact_lookup_after_false_updates',24),
                           ('exact_lookup_after_corrections',32),('exact_lookup_source_filtered',32)]:
            assert run['conditions'][name]['episode_correct']==[count]*len(truth)
        assert all(run[k] for k in ['controller_unchanged','user_state_isolation','reset_replay_exact','memory_weights_changed'])
        for metric in run['targeted']:
            targeted.setdefault(metric,[0,0])
            targeted[metric][0]+=run['targeted'][metric]['correct'];targeted[metric][1]+=run['targeted'][metric]['total']
        for load,stats in run['capacity'].items():
            assert stats['correct']==sum(stats['episode_correct'])
            assert stats['total']==int(load)*r['protocol']['test_episodes']
    for name,(count,den) in aggregates.items():
        s=r['summary']['conditions'][name]
        assert s['correct']==count and s['total']==den and abs(s['accuracy']-count/den)<1e-6
    for name,(count,den) in targeted.items():
        assert abs(r['summary']['targeted'][name]-count/den)<1e-6
    for load,mean in r['summary']['capacity'].items():
        exact=sum(run['capacity'][load]['correct'] for run in r['runs'])/sum(run['capacity'][load]['total'] for run in r['runs'])
        assert abs(exact-mean)<1e-6
    report=dict(passed=True,audited_neural_memory_predictions=total,seeds=len(r['runs']),
                heldout_streams_per_seed=r['protocol']['test_episodes'],
                same_test_streams_reused_across_seeds=True,prediction_sha256=hashes)
    (root/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
