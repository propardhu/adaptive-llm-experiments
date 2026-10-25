"""A Titans-inspired associative neural-memory study, not a Titans LLM replica.

Only a separate two-layer memory changes during evaluation. The controller is
meta-trained on clean streams. The study uses random vector keys and eight
symbolic values; it does not evaluate natural language or a pretrained LLM.
"""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

RAW = 32
KEY = 32
HIDDEN = 64
VALUES = 8
SEEDS = [11, 23, 37]


def tensor_digest(named):
    h = hashlib.sha256()
    for name, tensor in named:
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def state_digest(state):
    return tensor_digest(zip(['w1', 'w2', 'momentum1', 'momentum2'], state))


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(RAW, KEY, bias=False)
        nn.init.orthogonal_(self.key.weight)
        self.gate = nn.Linear(RAW + VALUES, 3)
        nn.init.zeros_(self.gate.weight)
        self.gate.bias.data.copy_(torch.tensor([0.0, -3.0, -5.0]))
        self.initial_w1 = nn.Parameter(torch.randn(HIDDEN, KEY))
        self.initial_w2 = nn.Parameter(torch.zeros(VALUES, HIDDEN))
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def initial_state(self, batch, training=False):
        w1 = self.initial_w1.unsqueeze(0).expand(batch, -1, -1)
        w2 = self.initial_w2.unsqueeze(0).expand(batch, -1, -1)
        if not training:
            w1 = w1.detach().clone().requires_grad_()
            w2 = w2.detach().clone().requires_grad_()
        return (w1, w2, torch.zeros_like(w1), torch.zeros_like(w2))

    def prepare(self, raw, value, training):
        keys = F.normalize(self.key(raw), dim=-1)
        target = F.one_hot(value, VALUES).to(raw.dtype)
        gates = self.gate(torch.cat([raw, target], dim=-1)).sigmoid()
        # Learned, input-dependent controls. Bounds are fixed before testing.
        eta = 2.0 * gates[..., 0]
        beta = 0.9 * gates[..., 1]
        alpha = 0.02 * gates[..., 2]
        if not training:
            keys, eta, beta, alpha = [t.detach() for t in [keys, eta, beta, alpha]]
        return keys, target, eta, beta, alpha

    def write(self, state, raw, value, training=False):
        w1, w2, s1, s2 = state
        key, target, eta, beta, alpha = self.prepare(raw, value, training)
        prediction = memory_read(w1, w2, key[:, None, :]).squeeze(1)
        # Sum independent episodes so batch size does not change each update.
        loss = 0.5 * (prediction - target).square().sum()
        g1, g2 = torch.autograd.grad(loss, (w1, w2), create_graph=training)
        eta, beta, alpha = [x[:, None, None] for x in [eta, beta, alpha]]
        ns1, ns2 = beta * s1 - eta * g1, beta * s2 - eta * g2
        next_state = ((1.0 - alpha) * w1 + ns1, (1.0 - alpha) * w2 + ns2, ns1, ns2)
        if not training:
            next_state = tuple(x.detach().requires_grad_(i < 2) for i, x in enumerate(next_state))
        return next_state

    def read(self, state, raw, training=False):
        keys = F.normalize(self.key(raw), dim=-1)
        scale = self.log_scale.exp().clamp(max=100)
        if not training:
            keys, scale = keys.detach(), scale.detach()
        return scale * memory_read(state[0], state[1], keys)


def memory_read(w1, w2, keys):
    hidden = torch.tanh(torch.einsum('bkd,bhd->bkh', keys, w1))
    hidden = F.normalize(hidden, dim=-1)
    return torch.einsum('bkh,bvh->bkv', hidden, w2)


def episodes(batch, count, rng):
    assert count % VALUES == 0
    raw = F.normalize(torch.randn(batch, count, RAW, generator=rng), dim=-1)
    balanced = torch.arange(VALUES).repeat(count // VALUES)
    values = torch.stack([balanced[torch.randperm(count, generator=rng)] for _ in range(batch)])
    return raw, values


def write_stream(model, raw, values, training=False, state=None):
    state = model.initial_state(raw.shape[0], training) if state is None else state
    for t in range(raw.shape[1]):
        state = model.write(state, raw[:, t], values[:, t], training=training)
    return state


def prediction(model, state, raw):
    with torch.no_grad():
        logits = model.read(state, raw)
    return logits.argmax(-1), logits


def scores(pred, target, mask=None):
    correct = pred.eq(target)
    if mask is None:
        mask = torch.ones_like(correct, dtype=torch.bool)
    counts = (correct & mask).sum(-1)
    den = mask.sum(-1)
    return dict(correct=int(counts.sum()), total=int(den.sum()),
                accuracy=float(counts.sum() / den.sum()),
                episode_correct=counts.tolist(), episode_total=den.tolist())


def train(seed, protocol, out):
    torch.manual_seed(seed)
    model = Controller()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol['outer_lr'])
    rng = torch.Generator().manual_seed(10000 + seed)
    dev_raw, dev_values = episodes(128, protocol['train_facts'], torch.Generator().manual_seed(50000))
    history = []
    started = time.perf_counter()
    for step in range(1, protocol['outer_steps'] + 1):
        raw, target = episodes(protocol['outer_batch'], protocol['train_facts'], rng)
        state = write_stream(model, raw, target, training=True)
        logits = model.read(state, raw, training=True)
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % protocol['log_every'] == 0:
            dev_state = write_stream(model, dev_raw, dev_values)
            pred, _ = prediction(model, dev_state, dev_raw)
            item = dict(step=step, loss=float(loss.detach()), development_recall=scores(pred, dev_values)['accuracy'])
            history.append(item)
            print(json.dumps(dict(seed=seed, **item)), flush=True)
    saved = out / f'controller-{seed}.pt'
    torch.save(dict(model=model.state_dict(), initial_model=initial, protocol=protocol), saved)
    return model, initial, dict(seconds=time.perf_counter()-started, history=history)


def evaluate(seed, model, initial, protocol, out):
    before = tensor_digest(model.named_parameters())
    count, batch = protocol['test_facts'], protocol['test_episodes']
    rng = torch.Generator().manual_seed(protocol['test_data_seed'])
    raw, target = episodes(batch, count, rng)
    clean = write_stream(model, raw, target)
    clean_pred, clean_logits = prediction(model, clean, raw)
    empty_pred, _ = prediction(model, model.initial_state(batch), raw)
    untrained = Controller()
    untrained.load_state_dict(initial)
    random_state = write_stream(untrained, raw, target)
    random_pred, _ = prediction(untrained, random_state, raw)
    k = count // 4
    indices = torch.stack([torch.randperm(count, generator=rng)[:k] for _ in range(batch)])
    rows = torch.arange(batch)[:, None]
    bad_raw = raw[rows, indices]
    correct_values = target[rows, indices]
    wrong_values = (correct_values + torch.randint(1, VALUES, correct_values.shape, generator=rng)) % VALUES
    bad_mask = torch.zeros_like(target, dtype=torch.bool)
    bad_mask[rows, indices] = True
    poison = write_stream(model, bad_raw, wrong_values, state=clean)
    poison_pred, _ = prediction(model, poison, raw)
    corrected = write_stream(model, bad_raw, correct_values, state=poison)
    correction_pred, _ = prediction(model, corrected, raw)
    # Source allowlist uses supplied provenance, not a learned truth detector.
    # Rejected documents make no write, no momentum update and no decay step.
    gated = clean
    gated_pred, _ = prediction(model, gated, raw)
    replay = write_stream(model, raw, target)
    replay_pred, replay_logits = prediction(model, replay, raw)
    assert torch.equal(clean_logits, replay_logits)
    assert state_digest(clean) == state_digest(replay)

    conditions = {name: scores(pred, target) for name, pred in [
        ('no_write', empty_pred), ('untrained_memory', random_pred),
        ('clean_memory', clean_pred), ('after_false_updates', poison_pred),
        ('after_verified_corrections', correction_pred),
        ('source_filtered', gated_pred), ('reset_and_replay', replay_pred)]}
    conditions['exact_lookup_clean'] = scores(target, target)
    lookup_poison = target.clone(); lookup_poison[rows, indices] = wrong_values
    conditions['exact_lookup_after_false_updates'] = scores(lookup_poison, target)
    conditions['exact_lookup_after_corrections'] = scores(target, target)
    conditions['exact_lookup_source_filtered'] = scores(target, target)
    targeted = dict(
        clean=scores(clean_pred, target, bad_mask),
        poisoned=scores(poison_pred, target, bad_mask),
        corrected=scores(correction_pred, target, bad_mask),
        poison_follows_false_value=scores(poison_pred[rows, indices], wrong_values),
        clean_other_facts=scores(clean_pred, target, ~bad_mask),
        poisoned_other_facts=scores(poison_pred, target, ~bad_mask))

    stress = {}
    for load in protocol['memory_loads']:
        xr, yr = episodes(batch, load, torch.Generator().manual_seed(protocol['capacity_seed'] + load))
        st = write_stream(model, xr, yr)
        pp, _ = prediction(model, st, xr)
        stress[str(load)] = scores(pp, yr)

    # Separate users have separate fast state; writing B cannot alter A.
    first = tuple(t[:1].detach().clone().requires_grad_(i < 2) for i, t in enumerate(clean))
    a_before = state_digest(first)
    b = model.initial_state(1)
    b = write_stream(model, bad_raw[1:2], wrong_values[1:2], state=b)
    assert state_digest(first) == a_before
    assert tensor_digest(model.named_parameters()) == before
    check_path = out / f'session-{seed}.pt'
    torch.save(dict(state=[t.detach() for t in first], raw=raw[:1], target=target[:1],
                    logits=clean_logits[:1], controller_sha256=before), check_path)
    predictions = dict(target=target.tolist(), attacked_indices=indices.tolist(), false_values=wrong_values.tolist(),
                       conditions={name: pred.tolist() for name, pred in [
                       ('no_write',empty_pred),('untrained_memory',random_pred),('clean_memory',clean_pred),
                       ('after_false_updates',poison_pred),('after_verified_corrections',correction_pred),
                       ('source_filtered',gated_pred),('reset_and_replay',replay_pred)]})
    (out / f'predictions-{seed}.json').write_text(json.dumps(predictions, separators=(',', ':')))
    return dict(seed=seed, conditions=conditions, targeted=targeted, capacity=stress,
                controller_sha256=before, controller_unchanged=True, user_state_isolation=True,
                reset_replay_exact=True, memory_weights_changed=state_digest(clean) != state_digest(model.initial_state(batch)),
                checkpoint=check_path.name)


def aggregate(runs):
    names = list(runs[0]['conditions'])
    conditions = {}
    for name in names:
        rows = [r['conditions'][name] for r in runs]
        values = [r['accuracy'] for r in rows]
        conditions[name] = dict(accuracy=sum(values)/len(values), min=min(values), max=max(values),
                                correct=sum(r['correct'] for r in rows), total=sum(r['total'] for r in rows))
    return dict(conditions=conditions,
                capacity={n: sum(r['capacity'][n]['accuracy'] for r in runs)/len(runs) for n in runs[0]['capacity']},
                targeted={n: sum(r['targeted'][n]['accuracy'] for r in runs)/len(runs) for n in runs[0]['targeted']})


def smoke():
    torch.manual_seed(123)
    model = Controller().double()
    raw, target = episodes(2, 8, torch.Generator().manual_seed(1))
    raw=raw.double()
    initial=model.initial_state(2)
    before=tensor_digest(model.named_parameters())
    updated=model.write(initial,raw[:,0],target[:,0])
    single=model.write(model.initial_state(1),raw[:1,0],target[:1,0])
    assert all(torch.allclose(a[:1],b,atol=1e-12) for a,b in zip(updated,single))
    assert tensor_digest(model.named_parameters())==before
    assert state_digest(initial)!=state_digest(updated)
    # Independent finite-difference check of the write loss derivatives.
    k=F.normalize(model.key(raw[:1,0]),dim=-1).detach()[:,None,:]
    v=F.one_hot(target[:1,0],VALUES).double()
    w1=torch.randn(1,3,KEY,dtype=torch.double,requires_grad=True)*.1
    w2=torch.randn(1,VALUES,3,dtype=torch.double,requires_grad=True)*.1
    assert torch.autograd.gradcheck(lambda a,b:.5*(memory_read(a,b,k).squeeze(1)-v).square().sum(),(w1,w2))
    training_state=write_stream(model,raw,target,training=True)
    loss=F.cross_entropy(model.read(training_state,raw,training=True).flatten(0,1),target.flatten())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(json.dumps(dict(finite_difference_gradient_check=True,batch_independence=True,
                         frozen_controller_check=True,meta_gradient_check=True)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='results')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--steps',type=int,default=1000)
    ap.add_argument('--episodes',type=int,default=200)
    args=ap.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if args.smoke:return smoke()
    out=Path(args.output)
    if (out/'results.json').exists():raise SystemExit('Use a new output directory to preserve recorded results.')
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(seeds=SEEDS,raw_dimension=RAW,key_dimension=KEY,hidden_dimension=HIDDEN,value_classes=VALUES,
                  train_facts=16,test_facts=32,outer_steps=args.steps,outer_batch=32,outer_lr=.002,log_every=100,
                  test_episodes=args.episodes,test_data_seed=90001,capacity_seed=91001,memory_loads=[8,16,32,64],
                  false_update_fraction=.25,inner_updates_per_fact=1,inner_loss='half squared key-value association error',
                  outer_loss='cross entropy of recalled values after the complete clean stream',
                  selection='fixed final training step; no test-based selection or hyperparameter sweep',
                  source_gate='perfectly supplied source labels: clean/correction events trusted; injected false events untrusted; no truth classifier',
                  model_scope='Titans-inspired two-layer neural memory and meta-trained controller; no attention, natural-language model, or full Titans reproduction',
                  runtime=dict(python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),device='cpu',threads=1))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    runs=[];started=time.perf_counter()
    for seed in SEEDS:
        model,initial,training=train(seed,protocol,out)
        run=evaluate(seed,model,initial,protocol,out);run['training']=training;runs.append(run)
        print(json.dumps(dict(seed=seed,results={k:round(v['accuracy'],4) for k,v in run['conditions'].items()})),flush=True)
        (out/'progress.json').write_text(json.dumps(dict(protocol=protocol,runs=runs),indent=2))
    result=dict(protocol=protocol,runs=runs,summary=aggregate(runs),elapsed_seconds=time.perf_counter()-started,
                controller_parameter_count=sum(p.numel() for p in model.parameters()),
                per_session_memory_parameters=HIDDEN*KEY+VALUES*HIDDEN,
                per_session_state_bytes=(HIDDEN*KEY+VALUES*HIDDEN)*4*2)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    (out/'summary.json').write_text(json.dumps(result['summary'],indent=2))
    print(json.dumps(dict(complete=True,seconds=result['elapsed_seconds'],summary=result['summary'])),flush=True)


if __name__=='__main__':main()
