"""Unit checks of envs.commands.motion_tracking.reset.sgs.SuccessGuidedSampler (no simulator):
python tests/test_sgs.py  (from the repo root, FlashDexRetarget env)."""
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# load the module file directly: the package __init__ pulls in the simulator stack
_p = Path(__file__).resolve().parents[1] / "src/envs/commands/motion_tracking/reset/sgs.py"
_spec = importlib.util.spec_from_file_location("sgs", _p)
sgs_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sgs_mod)
SGS = sgs_mod.SuccessGuidedSampler


def cfg(**kw):
    base = dict(num_bins=4, history=5, target=0.5, kappa=1.0, temperature=2.0, eps=1e-4, success="tracking",
                grace_steps=2, obj_trans=0.03, obj_rot_deg=30.0)
    base.update(kw)
    return SimpleNamespace(**base)


torch.manual_seed(0)
frames = torch.tensor([100, 400, 50])
B, S = 64, 2

# ── 1. empty table -> uniform over configurations -> uniform clip, start uniform in [0, 0.99 T) ──
s = SGS(frames, B, S, cfg(), "cpu")
assert s.n == 12 and torch.allclose(s.probs(), torch.full((12,), 1 / 12))
n = 120_000
clips, starts = [], []
for k in range(n // B):
    c, st = s.draw(torch.arange(B))
    clips.append(c), starts.append(st)
c, st = torch.cat(clips), torch.cat(starts)
frac = torch.bincount(c, minlength=3).float() / len(c)
assert (frac - 1 / 3).abs().max() < 0.01, frac
for ci in range(3):
    x = st[c == ci].float() / (0.99 * frames[ci])
    assert x.min() >= 0 and x.max() < 1.0 + 1e-6
    hist = torch.histc(x, bins=10, min=0.0, max=1.0) / len(x)
    assert (hist - 0.1).abs().max() < 0.012, (ci, hist)          # uniform start frame
print("1. empty table: uniform configurations -> uniform clip and uniform start in [0, 0.99 T)  OK")

# ── 2. weights = (p+eps)^(kt) (1-p+eps)^(k(1-t)), l = log(w+eps), P = softmax(l/T) ──
s = SGS(frames, B, S, cfg(history=4), "cpu")
pattern = {0: [1, 1, 1, 1], 1: [1, 0, 1, 0], 2: [0, 0, 0, 0], 3: [1, 0, 0, 0]}
for i, outs in pattern.items():
    s.outcomes[i, :4] = torch.tensor(outs, dtype=torch.bool)
    s.counts[i] = 4
p = s.success_rates()
assert torch.allclose(p[:4], torch.tensor([1.0, 0.5, 0.0, 0.25])) and float(p[4:].sum()) == 0.0
eps, t, kap, T = 1e-4, 0.5, 1.0, 2.0
w = (p + eps) ** (kap * t) * (1 - p + eps) ** (kap * (1 - t))
ref = torch.softmax(torch.log(w + eps) / T, dim=0)
assert torch.allclose(s.probs(), ref, atol=1e-7)
pr = s.probs()
assert pr[1] > pr[3] > pr[0] and abs(float(pr[0] - pr[2])) < 1e-6   # mid > low > solved == unsolved/unseen
print(f"2. beta weights: P(p=0.5)/P(p=1) = {float(pr[1] / pr[0]):.2f}, matches Eq. 1-2  OK")

# ── 3. record: duplicate configurations in one batch fill the ring buffer in order; counts cap at H ──
s = SGS(frames, 8, S, cfg(history=3, success="completion"), "cpu")
env = torch.arange(8)
s.env_cfg[:] = torch.tensor([5, 5, 5, 5, 2, 2, 7, -1])
done = torch.tensor([1, 0, 1, 1, 0, 1, 1, 1], dtype=torch.bool)
succ = s.record(env, done)
assert succ.tolist() == [True, False, True, True, False, True, True]          # env 7 (-1) skipped
assert int(s.counts[5]) == 3 and int(s.counts[2]) == 2 and int(s.counts[7]) == 1 and int(s.counts.sum()) == 6
assert int(s.ptr[5]) == 4 % 3 and s.total_recorded == 7
assert s.outcomes[5].tolist() == [True, False, True]                         # ranks 0..3 -> slots 0,1,2,0
assert s.outcomes[2].tolist() == [False, True, False] and bool(s.outcomes[7, 0])
assert (s.env_cfg[:7] == -1).all()                                           # episodes closed
assert s.record(env, done).numel() == 0                                      # nothing open -> nothing recorded
print("3. ring buffers: duplicates in one batch, H cap, closed episodes ignored  OK")

# ── 4. tracking criterion: grace steps excluded; both hands' objects must pass; no valid step -> completion ──
s = SGS(frames, 4, 2, cfg(grace_steps=2), "cpu")
s.draw(torch.arange(4))
big = torch.tensor([[0.5, 0.5]] * 4)                    # huge errors during the grace window
small_t = torch.tensor([[0.01, 0.02], [0.04, 0.01], [0.01, 0.01], [0.0, 0.0]])
small_r = torch.tensor([[0.1, 0.1], [0.1, 0.1], [math.radians(35), 0.1], [0.0, 0.0]])
s.accumulate(big, torch.full((4, 2), 3.0))
s.accumulate(big, torch.full((4, 2), 3.0))
for _ in range(5):
    s.accumulate(small_t, small_r)
assert s.n_valid.tolist() == [5, 5, 5, 5]
succ = s.record(torch.arange(4), torch.tensor([True, True, True, False]))
assert succ.tolist() == [True, False, False, False], succ               # trans fail (side 0 of env 1), rot fail, not completed
s.draw(torch.arange(2))
s.accumulate(big, torch.full((4, 2), 3.0))                               # still inside the grace window
assert s.n_valid[:2].tolist() == [0, 0]
assert s.record(torch.arange(2), torch.tensor([True, False])).tolist() == [True, False]
c = SGS(frames, 2, 2, cfg(success="completion"), "cpu")
c.draw(torch.arange(2))
c.accumulate(big, torch.full((2, 2), 3.0))
assert c.record(torch.arange(2), torch.tensor([True, False])).tolist() == [True, False]   # errors ignored
print("4. tracking = completed + mean obj error <= 3 cm / 30 deg on every hand after the grace window; completion "
      "ignores errors  OK")

# ── 5. invalidate (eval takeover) and state round trip ──
s.draw(torch.arange(4))
s.invalidate()
assert s.record(torch.arange(4), torch.ones(4, dtype=torch.bool)).numel() == 0
st = s.state_dict()
s2 = SGS(frames, 4, 2, cfg(grace_steps=2), "cpu")
s2.load_state_dict(st)
assert torch.equal(s2.outcomes, s.outcomes) and torch.equal(s2.counts, s.counts) and torch.equal(s2.ptr, s.ptr)
assert torch.allclose(s2.probs(), s.probs()) and s2.total_recorded == s.total_recorded
try:
    SGS(frames, 4, 2, cfg(num_bins=8), "cpu").load_state_dict(st)
    raise AssertionError("mismatched state accepted")
except ValueError:
    pass
st_keys = sorted(s.stats())
assert "SGS/entropy_norm" in st_keys and "SGS/mean_success" in st_keys
print("5. invalidate drops running episodes; state_dict round trip exact; mismatched table refused; stats keys  OK")
print("ALL OK")
