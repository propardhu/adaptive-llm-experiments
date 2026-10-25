"""Controlled rerun with internal adapters, development selection and retention."""
import argparse
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import *


class Delta(nn.Module):
    def __init__(self, width, output, rank=8):
        super().__init__()
        self.a=nn.Parameter(torch.randn(rank,width)*0.02)
        self.b=nn.Parameter(torch.zeros(output,rank))
    def forward(self,x):return F.linear(F.linear(x,self.a),self.b)


class Wrapped(nn.Module):
    def __init__(self, base, delta):
        super().__init__();self.base=base;self.delta=delta
    def forward(self,x):return self.base(x)+self.delta(x)


def digest(pairs):
    d=hashlib.sha256()
    for name,p in pairs:d.update(name.encode());d.update(p.detach().contiguous().cpu().numpy().tobytes())
    return d.hexdigest()


class Study:
    def __init__(self, proto, device):
        self.p=proto;self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(proto['model'],revision=proto['revision'])
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model=AutoModelForCausalLM.from_pretrained(proto['model'],revision=proto['revision'],
                        torch_dtype=torch.float32,attn_implementation='eager').to(device).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.base_pairs=list(self.model.named_parameters());self.base_hash=digest(self.base_pairs)
        self.layers=self.model.model.layers[-proto['internal_layers']:]
        self.head_delta=None;self.deltas=nn.ModuleList()

    def token(self,word):
        ids=self.tokenizer.encode(' '+word,add_special_tokens=False)
        assert len(ids)==1,(word,ids)
        return ids[0]

    @torch.no_grad()
    def cache(self,texts):
        enc=self.tokenizer(texts,padding=True,return_tensors='pt',add_special_tokens=False).to(self.device)
        inputs=[];masks=[];outputs=[];capture={}
        def hook(module,args,kwargs):
            capture['h']=args[0].detach();capture['kw']=kwargs
        handle=self.layers[0].register_forward_pre_hook(hook,with_kwargs=True)
        for start in range(0,len(texts),8):
            batch={k:v[start:start+8] for k,v in enc.items()}
            h=self.model.model(**batch,use_cache=False).last_hidden_state
            lens=batch['attention_mask'].sum(1)-1
            outputs.append(h[torch.arange(len(h),device=self.device),lens].detach())
            inputs.append(capture['h']);masks.append(capture['kw']['attention_mask'])
        handle.remove()
        kw={k:v for k,v in capture['kw'].items() if k!='attention_mask'}
        result={'texts':texts,'input':torch.cat(inputs),'mask':torch.cat(masks),
                'lengths':enc.attention_mask.sum(1)-1,'kwargs':kw,'final':torch.cat(outputs)}
        result['base_logits']=self.model.lm_head(result['final'])
        # Verify the cached execution path before any adaptation is applied.
        recovered=self.logits(result,torch.arange(min(4,len(texts)),device=self.device),'internal_adapter')
        assert torch.allclose(recovered,result['base_logits'][:len(recovered)],atol=2e-5,rtol=2e-5)
        return result

    def setup(self,method,seed):
        self.clear();torch.manual_seed(seed)
        if method=='output_adapter':
            self.head_delta=Delta(self.model.config.hidden_size,self.model.config.vocab_size,self.p['rank']).to(self.device)
            self.deltas=nn.ModuleList([self.head_delta])
        else:
            ds=[]
            for layer in self.layers:
                base=layer.mlp.down_proj
                delta=Delta(base.in_features,base.out_features,self.p['rank']).to(self.device)
                layer.mlp.down_proj=Wrapped(base,delta);ds.append(delta)
            self.deltas=nn.ModuleList(ds)

    def clear(self):
        for layer in self.layers:
            if isinstance(layer.mlp.down_proj,Wrapped):layer.mlp.down_proj=layer.mlp.down_proj.base
        self.head_delta=None;self.deltas=nn.ModuleList()

    def logits(self,pool,idx,method):
        if method=='output_adapter':
            return pool['base_logits'][idx]+self.head_delta(pool['final'][idx])
        h=pool['input'][idx]
        for layer in self.layers:h=layer(h,attention_mask=pool['mask'][idx],**pool['kwargs'])[0]
        h=self.model.model.norm(h)
        h=h[torch.arange(len(idx),device=self.device),pool['lengths'][idx]]
        return self.model.lm_head(h)

    @torch.no_grad()
    def evaluate(self,pool,method=None,targets=None):
        logits=[]
        for start in range(0,len(pool['texts']),12):
            idx=torch.arange(start,min(start+12,len(pool['texts'])),device=self.device)
            logits.append(pool['base_logits'][idx] if method is None else self.logits(pool,idx,method))
        z=torch.cat(logits);base=pool['base_logits'];pred=z.argmax(-1)
        logp=z.log_softmax(-1)
        items=[]
        for i,prompt in enumerate(pool['texts']):
            record=dict(prompt=prompt,prediction=self.tokenizer.decode([pred[i].item()]),
                        base_prediction=self.tokenizer.decode([base[i].argmax().item()]),
                        agreement=bool(pred[i]==base[i].argmax()))
            if targets is not None:
                record.update(target=self.tokenizer.decode([targets[i].item()]),
                              correct=bool(pred[i]==targets[i]),
                              target_probability=float(logp[i,targets[i]].exp()))
            items.append(record)
        kl=F.kl_div(logp,base.softmax(-1),reduction='batchmean').item()
        return dict(n=len(items),correct=None if targets is None else sum(x['correct'] for x in items),
                    agreement=sum(x['agreement'] for x in items),kl_from_base=max(0.,kl),items=items)

    def prepare(self,split):
        fs=facts(split)
        train=prompts(fs,TRAIN_TEMPLATES)
        ev=prompts(fs,DEV_TEMPLATES if split=='dev' else TEST_TEMPLATES)
        retention=DEV_RETENTION if split=='dev' else TEST_RETENTION
        print('Caching',split,'train',len(train),'evaluation',len(ev),'retention',len(retention),flush=True)
        return dict(train=self.cache([x[0] for x in train]),
                    train_ids=torch.tensor([self.token(x[1]) for x in train],device=self.device),
                    eval=self.cache([x[0] for x in ev]),
                    eval_ids=torch.tensor([self.token(x[1]) for x in ev],device=self.device),
                    retention=self.cache(retention))

    def run(self,pools,anchor,method,lr,kl_weight,seed,steps,checkpoints,match_anchor_draws=False):
        self.setup(method,seed)
        optimizer=torch.optim.Adam(self.deltas.parameters(),lr=lr)
        rng=torch.Generator().manual_seed(seed+44)
        snapshots=[];curve=[];elapsed=0.
        for step in range(1,steps+1):
            tick=time.perf_counter()
            idx=torch.randint(len(pools['train']['texts']),(self.p['batch_size'],),generator=rng).to(self.device)
            z=self.logits(pools['train'],idx,method)
            loss=F.cross_entropy(z,pools['train_ids'][idx])
            kl=torch.zeros((),device=self.device)
            if kl_weight:
                aidx=torch.randint(len(anchor['texts']),(self.p['batch_size'],),generator=rng).to(self.device)
                az=self.logits(anchor,aidx,method)
                kl=F.kl_div(az.log_softmax(-1),anchor['base_logits'][aidx].softmax(-1),reduction='batchmean')
            elif match_anchor_draws:
                # Consume the same random draw as the protected run, so the fact
                # minibatches match exactly in the follow-up ablation.
                torch.randint(len(anchor['texts']),(self.p['batch_size'],),generator=rng)
            optimizer.zero_grad(set_to_none=True)
            total=loss+kl_weight*kl;total.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(self.deltas.parameters(),1.0))
            optimizer.step();elapsed+=time.perf_counter()-tick
            curve.append(dict(step=step,fact_loss=float(loss.detach()),anchor_kl=float(kl.detach()),gradient_norm=norm))
            if step in checkpoints:
                ev=self.evaluate(pools['eval'],method,pools['eval_ids'])
                ret=self.evaluate(pools['retention'],method)
                score=ev['correct']/ev['n']-2*(1-ret['agreement']/ret['n'])
                snapshots.append(dict(step=step,facts=ev,retention=ret,selection_score=score,update_seconds=elapsed))
                print(f'{method} lr={lr} kl={kl_weight} seed={seed} step={step}: facts={ev["correct"]}/{ev["n"]} retention={ret["agreement"]}/{ret["n"]}',flush=True)
        return dict(method=method,lr=lr,kl_weight=kl_weight,seed=seed,
                    trainable_parameters=sum(p.numel() for p in self.deltas.parameters()),
                    checkpoints=snapshots,curve=curve)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='results')
    ap.add_argument('--device',default='cpu',choices=['cpu','cuda','mps'])
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();torch.set_num_threads(2)
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    p=protocol()
    if args.smoke:p.update(steps=2,checkpoints=[2],learning_rates=[.001],protected_kl_weights=[1.],final_seeds=[7])
    (out/'protocol.json').write_text(json.dumps(p,indent=2))
    study=Study(p,args.device)
    anchor=study.cache(ANCHORS)
    dev=study.prepare('dev')
    development=[];selected={}
    for method in p['methods']:
        weights=p['protected_kl_weights'] if method=='internal_adapter_kl' else [0.]
        candidates=[]
        for lr,weight in itertools.product(p['learning_rates'],weights):
            r=study.run(dev,anchor,method,lr,weight,p['development_seed'],p['steps'],p['checkpoints'])
            development.append(r)
            for cp in r['checkpoints']:
                candidates.append(dict(method=method,lr=lr,kl_weight=weight,steps=cp['step'],
                                       selection_score=cp['selection_score'],dev_fact_correct=cp['facts']['correct'],
                                       dev_retention_agreement=cp['retention']['agreement']))
            (out/'development.json').write_text(json.dumps(development,indent=2))
        best=max(candidates,key=lambda c:(c['selection_score'],-c['steps'],-c['lr'],c['kl_weight']))
        selected[method]=best
    # Selection is locked before ANY evaluation on the final facts or retention set.
    (out/'selected.json').write_text(json.dumps(selected,indent=2))
    print('Development selection locked:',selected,flush=True)
    study.clear();del dev
    test=study.prepare('test')
    baseline=dict(facts=study.evaluate(test['eval'],targets=test['eval_ids']),
                  retention=study.evaluate(test['retention']))
    # Perfect relevant-fact selection, not an implemented retrieval system.
    fs=facts('test');ev=prompts(fs,TEST_TEMPLATES)
    oracle=[fs[i//len(TEST_TEMPLATES)]['text']+'\n'+q for i,(q,_) in enumerate(ev)]
    oracle_pool=study.cache(oracle)
    baseline['relevant_fact']=study.evaluate(oracle_pool,targets=test['eval_ids'])
    del oracle_pool
    results=[];persistence={}
    for method in p['methods']:
        cfg=selected[method]
        for seed in p['final_seeds']:
            r=study.run(test,anchor,method,cfg['lr'],cfg['kl_weight'],seed,cfg['steps'],[cfg['steps']])
            if method=='internal_adapter_kl' and seed==p['final_seeds'][0]:
                # Save trained adapter tensors, destroy them, then construct and reload.
                checkpoint={k:v.detach().cpu() for k,v in study.deltas.state_dict().items()}
                path=out/'session_adapter.pt';torch.save(checkpoint,path)
                before=r['checkpoints'][-1]['facts']
                study.setup(method,seed+500)
                study.deltas.load_state_dict(torch.load(path,map_location=study.device,weights_only=True))
                after=study.evaluate(test['eval'],method,test['eval_ids'])
                persistence=dict(reload_matches=before==after,checkpoint_bytes=path.stat().st_size,
                                 scope='serialize, destroy, reconstruct and reload adapter in one process; no cross-user service')
                assert before==after
            study.clear()
            reset=study.evaluate(test['eval'],targets=test['eval_ids'])
            r['reset_matches']=reset==baseline['facts'];assert r['reset_matches']
            results.append(r)
            (out/'test_partial.json').write_text(json.dumps(results,indent=2))
    unchanged=digest(study.base_pairs)==study.base_hash;assert unchanged
    result=dict(protocol=p,selection=selected,baseline=baseline,runs=results,persistence=persistence,
                base_unchanged=unchanged,base_sha256=study.base_hash,
                environment=dict(python=platform.python_version(),torch=torch.__version__,
                                 transformers=transformers.__version__,device=args.device,threads=2),
                note='Only final test evaluation after locked development selection is reported. Cached prefix is a speed optimization, not a serving benchmark.')
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print('Complete: frozen base, cache equivalence, reset and reload checks passed.',flush=True)


if __name__=='__main__':main()
