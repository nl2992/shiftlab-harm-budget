"""Post hoc ablations on the real streams (not part of protocol 05F), paired with run_real.py.

Same tapes, episodes, policy seeds and arm specs as the real-stream study, one change per arm:
  _labelblind  full sampling; the shield never credits a returned label (every exposed round stays
               booked at 1). The inner rule still learns from the labels.
  _noride      ride-out switched off (full and paced).
Writes outputs/paper3/riskbudget/real_ablations/<condition>.json: per stream and arm, the mean J
over episodes and seeds, and the number of runs with sup harm above R. The unchanged arms are read
from outputs/paper3/riskbudget/real, after checking that one episode per stream reproduces exactly.
"""

import json
from multiprocessing import Pool

import numpy as np

from shiftlab.paper3.riskbudget.harness import ROOT
from shiftlab.paper3.riskbudget.metrics import harm_summary
from shiftlab.paper3.riskbudget.real_streams import (
    CONDITIONS,
    HS,
    STREAMS,
    build_real_arms,
    corruption_rng,
    delay_of,
    load_stream,
    make_real_arm,
    make_real_spec,
    make_real_tape,
    run_arm_row,
)
from shiftlab.paper3.simulation import run_policy, seeds_for

BASE = ROOT / "outputs" / "paper3" / "riskbudget" / "real"
OUT = ROOT / "outputs" / "paper3" / "riskbudget" / "real_ablations"
SEEDS = seeds_for("dev", 20, 60000)
VARIANTS = [("f3_pi1", "labelblind"), ("f3_pi1", "noride"), ("f3v2d", "noride")]


class _NoCredit(dict):
    def get(self, *args):  # the full-sampling shield looks released labels up with .get
        return None


def make_ablation(spec: dict, delay: int, tag: str):
    pol = make_real_arm(spec, delay)
    if tag == "noride":
        assert hasattr(pol, "ride_out")
        pol.ride_out = False
        return pol
    reset = pol.reset

    def reset_blind(problem, rng):
        reset(problem, rng)
        pol.exposed = _NoCredit()

    pol.reset = reset_blind
    return pol


def _task(args):
    cond, stream = args
    arms = {a["name"]: a for a in build_real_arms()}
    base = json.loads((BASE / cond / f"{stream}.json").read_text())
    l0, l1 = load_stream(stream)
    delay = delay_of(stream)
    spec = make_real_spec(delay, cond)
    tapes = [
        make_real_tape(l0[e], l1[e], delay, cond, corruption_rng(stream, e, cond))
        for e in range(len(l0))
    ]
    out = {}
    for family, tag in VARIANTS:
        for h in HS:
            name = f"{family}_h{h}"
            arm = arms[name]
            # pairing check: the unchanged arm must reproduce the committed result
            i = base["arms"].index(name)
            for k, seed in enumerate(SEEDS):
                row = run_arm_row(arm, tapes[0], spec, delay, seed)
                assert row["J"] == base["fields"]["J"][i][0][k], (cond, stream, name, seed)
            J, viol, rode = [], 0, []
            for tape in tapes:
                for seed in SEEDS:
                    pol = make_ablation(arm, delay, tag)
                    rec = run_policy(pol, tape, spec, seed)
                    J.append(rec.J)
                    viol += harm_summary(tape, rec, spec.costs.k)["sup_harm"] > arm["R"] + 1e-9
                    rode.append(float(getattr(pol, "rode_out", 0)))
            out[f"{name}_{tag}"] = {
                "J": float(np.mean(J)),
                "n_viol": int(viol),
                "n_runs": len(J),
                "rode_out": float(np.mean(rode)),
            }
    return cond, stream, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {c: {} for c in CONDITIONS}
    with Pool(6) as pool:
        tasks = [(c, s) for c in CONDITIONS for s in STREAMS]
        for cond, stream, out in pool.imap_unordered(_task, tasks):
            res[cond][stream] = out
            print(f"done {cond}/{stream}", flush=True)
    for cond, by_stream in res.items():
        body = {s: by_stream[s] for s in STREAMS}
        (OUT / f"{cond}.json").write_text(json.dumps(body, indent=1) + "\n")


if __name__ == "__main__":
    main()
