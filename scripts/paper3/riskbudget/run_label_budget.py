"""Post hoc diagnostic (not part of protocol 05F): how the label budget limits the shield on the
real streams, clean feedback, h=50.

Reruns two shield arms and Wald with cutoff at the protocol's B=200 and with B=T, on the same tapes
and policy seeds. At B=200 every run must reproduce the committed cost exactly. Also records when
labels run out and how many forced rollbacks fall before and after.
Writes outputs/paper3/riskbudget/label_budget/clean.json.
"""
import json
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
from shiftlab.paper3.riskbudget.harness import ROOT
from shiftlab.paper3.riskbudget.real_streams import (
    STREAMS,
    build_real_arms,
    corruption_rng,
    delay_of,
    load_stream,
    make_real_arm,
    make_real_spec,
    make_real_tape,
)
from shiftlab.paper3.simulation import run_policy, seeds_for

SEEDS = seeds_for("dev", 20, 60000)
ARMS = ["f3_pi1_h50", "f3v2d_h50", "wald_cut"]
BASE = ROOT / "outputs/paper3/riskbudget/real/clean"
OUT = ROOT / "outputs/paper3/riskbudget/label_budget"

def task(stream):
    arms = {a["name"]: a for a in build_real_arms()}
    base = json.loads((BASE / f"{stream}.json").read_text())
    l0, l1 = load_stream(stream); delay = delay_of(stream)
    spec = make_real_spec(delay, "clean")
    tapes = [make_real_tape(l0[e], l1[e], delay, "clean", corruption_rng(stream, e, "clean")) for e in range(len(l0))]
    out = {}
    for name in ARMS:
        for tag, sp in (("B200", spec), ("Bfull", replace(spec, B=spec.T))):
            J, ex_t, fr_before, fr_after, dep_after, exhausted = [], [], 0, 0, [], 0
            for e, tape in enumerate(tapes):
                for k, seed in enumerate(SEEDS):
                    pol = make_real_arm(arms[name], delay)
                    log = []
                    act = pol.act
                    def logged(view, act=act, pol=pol, log=log):
                        d = act(view)
                        log.append((view.labels_remaining, getattr(pol, "n_forced_rollbacks", 0), d.action))
                        return d
                    pol.act = logged
                    rec = run_policy(pol, tape, sp, seed)
                    if tag == "B200":
                        assert rec.J == base["fields"]["J"][base["arms"].index(name)][e][k], (stream, name)
                    J.append(rec.J)
                    L, F, A = map(np.array, zip(*log))
                    z = np.flatnonzero(L <= 0)
                    total = F[-1]
                    if len(z):
                        i = z[0]; exhausted += 1; ex_t.append(i + 1)
                        before = F[i - 1] if i else 0
                        fr_before += before; fr_after += total - before
                        dep_after.append(float(A[i:].mean()))
                    else:
                        fr_before += total
            out[f"{name}_{tag}"] = dict(J=float(np.mean(J)), exhausted=exhausted / len(J),
                exhaust_t=float(np.median(ex_t)) if ex_t else None, forced_before=int(fr_before),
                forced_after=int(fr_after), dep_after=float(np.mean(dep_after)) if dep_after else None)
    out["never"] = float(np.mean([t.ell[:, 0].sum() for t in tapes]))
    return stream, out

def main():
    with Pool(8) as pool:
        res = dict(pool.map(task, STREAMS))
    OUT.mkdir(parents=True, exist_ok=True)
    body = {name: res[name] for name in STREAMS}
    (OUT / "clean.json").write_text(json.dumps(body, indent=1) + "\n")
    for key in [f"{a}_{t}" for a in ARMS for t in ("B200", "Bfull")]:
        print(key, round(float(np.mean([res[s][key]["J"] for s in STREAMS])), 2))


if __name__ == "__main__":
    main()
