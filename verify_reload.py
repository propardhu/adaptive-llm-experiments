"""Fresh-process verification of saved memory, controller identity and reset."""
import argparse
import json
from pathlib import Path
import torch
from experiment import Controller, prediction, tensor_digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='results');args=ap.parse_args()
    root=Path(args.results);torch.set_num_threads(1)
    checks=[]
    for seed in [11,23,37]:
        ckpt=torch.load(root/f'controller-{seed}.pt',map_location='cpu',weights_only=True)
        session=torch.load(root/f'session-{seed}.pt',map_location='cpu',weights_only=True)
        model=Controller();model.load_state_dict(ckpt['model'])
        state=tuple(t.requires_grad_(i<2) for i,t in enumerate(session['state']))
        pred,logits=prediction(model,state,session['raw'])
        assert torch.equal(logits,session['logits'])
        assert tensor_digest(model.named_parameters())==session['controller_sha256']
        _,empty=prediction(model,model.initial_state(1),session['raw'])
        assert not torch.equal(empty,logits)
        checks.append(dict(seed=seed,logits_exact=True,predictions_exact=True,controller_hash_matches=True,
                           reset_changes_memory_output=True,queries=pred.numel()))
    result=dict(fresh_process=True,checks=checks)
    (root/'reload_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
