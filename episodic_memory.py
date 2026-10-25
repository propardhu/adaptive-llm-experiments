"""A tiny gradient-memory experiment, NOT an LLM or a Titans/TTT reproduction.

Twelve symbols receive a new balanced six-class assignment in every episode.
A single gradient step writes those associations into temporary fast weights.
Outer training learns a symbol representation that makes the writes useful.
All test assignments are generated after training from a separate random seed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F

KEYS=12
VALUES=6
WIDTH=12


def assignments(batch,generator):
    # Exactly two of each value per episode; assignments change every time.
    base=torch.arange(VALUES).repeat(KEYS//VALUES)
    return torch.stack([base[torch.randperm(KEYS,generator=generator)] for _ in range(batch)])


def read_after_write(encoder,targets,create_graph=False,write=True):
    features=F.normalize(encoder,dim=-1)
    memory=torch.zeros(targets.shape[0],VALUES,WIDTH,requires_grad=True)
    if write:
        predicted=torch.einsum('bvd,kd->bkv',memory,features)
        expected=F.one_hot(targets,VALUES).float()
        # Sum across independent episodes; mean over each episode's twelve writes.
        loss=(.5*(predicted-expected).square().sum(-1).mean(-1)).sum()
        gradient=torch.autograd.grad(loss,memory,create_graph=create_graph)[0]
        memory=memory-KEYS*gradient
    return torch.einsum('bvd,kd->bkv',memory,features),memory


def evaluate(encoder,targets,write):
    # Autograd remains enabled: writing is an actual gradient step, even at test time.
    logits,memory=read_after_write(encoder.detach(),targets,write=write)
    pred=logits.argmax(-1)
    correct=(pred==targets).sum(-1)
    return dict(accuracy=float(correct.float().mean()/KEYS),
                total_correct=int(correct.sum()),total_predictions=int(targets.numel()),
                episode_correct=correct.tolist(),memory_l2=float(memory.detach().norm()),
                note='Recall of observed symbolic associations under novel assignments; no natural-language questions.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/memory')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    proto=dict(keys=KEYS,values=VALUES,width=WIDTH,seeds=[11,23,37],outer_steps=600,
               outer_batch=32,outer_lr=.01,inner_steps=1,inner_lr=KEYS,
               inner_objective='half squared association error, mean over 12 observed pairs',
               outer_objective='cross entropy of recalled values after the inner update',
               test_episodes=1000,test_seed=98765,
               caveat='144 shared encoder parameters and 72 temporary fast-memory entries per episode; not an LLM or a paper reproduction')
    (out/'protocol.json').write_text(json.dumps(proto,indent=2))
    results=[]
    for seed in proto['seeds']:
        torch.manual_seed(seed)
        encoder=torch.nn.Parameter(torch.randn(KEYS,WIDTH))
        initial=encoder.detach().clone()
        optimizer=torch.optim.Adam([encoder],lr=proto['outer_lr'])
        rng=torch.Generator().manual_seed(seed+400)
        train_hashes=set();history=[];start=time.perf_counter()
        for step in range(1,proto['outer_steps']+1):
            target=assignments(proto['outer_batch'],rng)
            for row in target:train_hashes.add(tuple(row.tolist()))
            logits,_=read_after_write(encoder,target,create_graph=True)
            loss=F.cross_entropy(logits.reshape(-1,VALUES),target.reshape(-1))
            optimizer.zero_grad();loss.backward();optimizer.step()
            if step%100==0:
                history.append(dict(step=step,loss=float(loss.detach())))
        test_rng=torch.Generator().manual_seed(proto['test_seed'])
        # Exclude exact assignments encountered during outer training.
        test=[];seen=set(train_hashes)
        while len(test)<proto['test_episodes']:
            row=assignments(1,test_rng)[0];key=tuple(row.tolist())
            if key not in seen:test.append(row);seen.add(key)
        targets=torch.stack(test)
        before=hashlib.sha256(encoder.detach().numpy().tobytes()).hexdigest()
        conditions={'untrained_encoder_with_write':evaluate(initial,targets,True),
                    'trained_encoder_no_write':evaluate(encoder,targets,False),
                    'trained_encoder_with_write':evaluate(encoder,targets,True)}
        after=hashlib.sha256(encoder.detach().numpy().tobytes()).hexdigest()
        assert before==after
        # Independently validate the gradient-based write against its algebraic value.
        _,written=read_after_write(encoder.detach(),targets[:3])
        closed=torch.einsum('bkv,kd->bvd',F.one_hot(targets[:3],VALUES).float(),F.normalize(encoder.detach(),dim=-1))
        assert torch.allclose(written,closed,atol=1e-6)
        result=dict(seed=seed,conditions=conditions,elapsed_seconds_including_evaluation=time.perf_counter()-start,
                    slow_encoder_unchanged_during_test=True,gradient_write_check=True,
                    test_assignments_disjoint_from_training=True,training_curve=history)
        results.append(result)
        print('seed',seed,{k:round(v['accuracy'],4) for k,v in conditions.items()},flush=True)
        torch.save({'encoder':encoder.detach(),'initial_encoder':initial},out/f'encoder-{seed}.pt')
    (out/'results.json').write_text(json.dumps(dict(protocol=proto,runs=results),indent=2))
    print('Synthetic memory experiment complete.',flush=True)


if __name__=='__main__':main()
