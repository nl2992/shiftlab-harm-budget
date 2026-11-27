"""Post hoc matched control for the no-width ablation (not part of registration 04F).

The registered arm f3_pi25_R12_noconf has no partner that differs only in the width. This adds
one: the same arm spec with use_conf_width left on. Both run on the registered regimes and
confirmation seeds, so runs are paired. The no-width arm is rerun and must reproduce the
registered per-regime violation counts and pooled cost exactly.
Writes outputs/paper3/riskbudget/noconf_control/summary.json.
"""

import copy
import json
from multiprocessing import Pool

import numpy as np

from shiftlab.paper3.riskbudget.harness import ROOT, all_regimes
from shiftlab.paper3.riskbudget.harness_v2 import make_arm_v2
from shiftlab.paper3.riskbudget.metrics import harm_summary
from shiftlab.paper3.simulation import (
    CONFIG_DIR,
    SimulationEvaluator,
    generate_tape,
    make_spec,
    run_policy,
    seeds_for,
)

OUT = ROOT / "outputs" / "paper3" / "riskbudget" / "noconf_control"
REGISTERED = (
    ROOT / "papers/paper3_cost_aware_adaptation/2026_programme/results/riskbudget"
) / "confirm_summary.json"
BASE, CONTROL, T = "f3_pi25_R12_noconf", "f3_pi25_R12_withconf", 300
SEEDS = seeds_for("confirmation", 1000, 500000)  # the registered block, reused for pairing
_REG = _ARMS = None


def arm_specs():
    arms = json.loads((CONFIG_DIR / "f3_confirmation_arms_v1.json").read_text())["arms"]
    base = next(a for a in arms if a["name"] == BASE)
    control = copy.deepcopy(base)
    control["name"] = CONTROL
    control["shield"]["use_conf_width"] = True
    return [base, control]


def _init():
    global _REG, _ARMS
    _REG, _ARMS = all_regimes(), arm_specs()


def _task(args):
    rid, seed = args
    reg = _REG[rid]
    tape, truth = generate_tape(reg, seed, T)
    spec = make_spec(reg, T)
    ev = SimulationEvaluator(tape, truth, spec.costs)
    rows = []
    for arm in _ARMS:
        pol = make_arm_v2(arm, T)
        rec = run_policy(pol, tape, spec, seed)
        rates = [e[0] for e in pol.exposed.values()]  # one entry per queried exposed round
        sup = harm_summary(tape, rec, spec.costs.k)["sup_harm"]
        rows.append(
            (
                ev.evaluate(rec)["J"],
                sup,
                float(np.mean(rec.a)),
                int(rec.q.sum()),
                sum(p < 1 for p in rates),
                len(rates),
            )
        )
    return seed, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registered = json.loads(REGISTERED.read_text())
    reg_arm = registered["C6"]["arms"][BASE]
    names = [BASE, CONTROL]
    per_regime, cells = {}, []
    with Pool(8, initializer=_init) as pool:
        for rid in all_regimes():
            res = dict(pool.imap_unordered(_task, [(rid, s) for s in SEEDS], chunksize=4))
            x = np.array([res[s] for s in SEEDS])  # seed, arm, (J, sup, deployed, labels, low, queried)
            cells.append(x)
            per_regime[rid] = {
                n: {
                    "viol": int((x[:, i, 1] > 12 + 1e-9).sum()),
                    "J": float(x[:, i, 0].mean()),
                    "sup_max": float(x[:, i, 1].max()),
                    "deployed": float(x[:, i, 2].mean()),
                }
                for i, n in enumerate(names)
            }
            assert per_regime[rid][BASE]["viol"] == reg_arm["per_regime"][rid]["k"], rid
            print("done", rid, flush=True)
    x = np.stack(cells)  # regime, seed, arm, field
    for col, key in ((0, "J"), (3, "labels")):
        assert abs(x[:, :, 0, col].mean() - registered["descriptive"][BASE][key]) < 1e-9
    diff = x[:, :, 1, 0] - x[:, :, 0, 0]  # control minus no-width, paired
    se = float(np.sqrt((diff.var(axis=1, ddof=1) / diff.shape[1]).sum()) / diff.shape[0])
    arms = {}
    for i, n in enumerate(names):
        worst = max(per_regime, key=lambda r: per_regime[r][n]["viol"])
        total = sum(c[n]["viol"] for c in per_regime.values())
        arms[n] = {
            "total_viol": total,
            "pooled_rate": total / x[:, :, i, 1].size,
            "worst": worst,
            "worst_viol": per_regime[worst][n]["viol"],
            "J": float(x[:, :, i, 0].mean()),
            "deployed": float(x[:, :, i, 2].mean()),
            "sup_max": float(x[:, :, i, 1].max()),
            "labels": float(x[:, :, i, 3].mean()),
            # share of queried exposed rounds that used the lower rate
            "low_rate_share": float(x[:, :, i, 4].sum() / x[:, :, i, 5].sum()),
        }
    summary = {
        "n_regimes": len(per_regime),
        "n_seeds": len(SEEDS),
        "R": 12,
        "dJ_control_minus_noconf": {"mean": float(diff.mean()), "se": se},
        "arms": arms,
        "per_regime": per_regime,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_regime"}, indent=1))


if __name__ == "__main__":
    main()
