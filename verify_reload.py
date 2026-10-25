"""Reconstruct the base model and restore the saved adapter in a new process."""
import argparse,json,os
from pathlib import Path
import torch
from experiment import Study,digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='results')
    args=ap.parse_args();out=Path(args.results)
    r=json.loads((out/'results.json').read_text());p=r['protocol']
    torch.set_num_threads(2);study=Study(p,'cpu');test=study.prepare('test')
    study.setup('internal_adapter_kl',999)
    study.deltas.load_state_dict(torch.load(out/'session_adapter.pt',map_location='cpu',weights_only=True))
    actual=study.evaluate(test['eval'],'internal_adapter_kl',test['eval_ids'])
    expected=next(x for x in r['runs'] if x['method']=='internal_adapter_kl' and x['seed']==p['final_seeds'][0])['checkpoints'][-1]['facts']
    assert actual==expected
    assert digest(study.base_pairs)==study.base_hash
    result=dict(fresh_process_reload_matches=True,probes=actual['n'],correct=actual['correct'],
                comparison='all stored per-probe predictions and target probabilities match exactly',
                scope='one saved session adapter restored over the same pinned base model; not continual multi-session learning')
    (out/'reload_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
