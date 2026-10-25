"""Post-primary matched ablation; no search or selection on final test scores."""
import argparse,json
from pathlib import Path
import torch
from experiment import Study,digest
from data import ANCHORS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='results')
    args=ap.parse_args();out=Path(args.results)
    prior=json.loads((out/'results.json').read_text());p=prior['protocol']
    cfg=prior['selection']['internal_adapter_kl']
    protocol=dict(description='Follow-up ablation added after the primary test run. No tuning. Match the protected recipe and remove only its retention penalty.',
                  learning_rate=cfg['lr'],steps=cfg['steps'],seeds=p['final_seeds'],
                  rank=p['rank'],internal_layers=p['internal_layers'],
                  same_fact_minibatches=True)
    (out/'matched_control_protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.set_num_threads(2);study=Study(p,'cpu');test=study.prepare('test')
    assert study.evaluate(test['eval'],targets=test['eval_ids'])==prior['baseline']['facts']
    runs=[]
    for seed in p['final_seeds']:
        r=study.run(test,{'texts':ANCHORS},'internal_adapter',cfg['lr'],0.,seed,cfg['steps'],[cfg['steps']],match_anchor_draws=True)
        runs.append(r);study.clear()
    assert digest(study.base_pairs)==study.base_hash
    (out/'matched_control.json').write_text(json.dumps(dict(protocol=protocol,runs=runs,base_unchanged=True),indent=2))
    print('Matched ablation complete.',flush=True)

if __name__=='__main__':main()
