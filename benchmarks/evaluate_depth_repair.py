"""Reproducible analytic benchmark; no private data or Smoother installation.

Run: uv run python benchmarks/evaluate_depth_repair.py --output /tmp/depth-eval
Truth denotes changes in particle identity, not changes in physical velocity.
The baselines intentionally share robust fit/cost primitives with the detector
so that the difference being measured is the model and decision rule.
"""

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from holod3_linker import repair_depth_links
from holod3_linker.depth_repair import _cost, _huber_fit, _noise


def cases(seed=260716, repeats=100, stress=False):
    rng = np.random.default_rng(seed)
    for kind in (
        "linear",
        "fast_linear",
        "accelerated",
        "helix",
        "velocity_change",
        "smooth_transition",
        "isolated_outliers",
        "burst_8",
        "burst_24",
        "burst_40",
        "endpoint_burst",
        "heteroscedastic",
        "step_12",
        "step_30",
        "step_100",
        "step_with_outliers",
        "step_gap",
        "two_steps",
        "long_gap",
    ):
        for rep in range(repeats):
            n = int(rng.integers(120, 360))
            t = np.arange(n, dtype=float)
            z = 100 + rng.uniform(-0.3, 0.3) * t
            truth = []
            sigma = rng.uniform(0.5, 2)
            if kind == "fast_linear":
                z = 100 + rng.choice([-1, 1]) * rng.uniform(1, 8) * t
            elif kind == "accelerated":
                z = 100 + 0.008 * (t - n / 2) ** 2
            elif kind == "helix":
                z = 100 + 80 * np.sin(t / rng.uniform(16, 40) + rng.uniform(0, 6))
            elif kind == "velocity_change":
                z = 100 + 0.1 * t + rng.uniform(0.5, 3) * np.maximum(t - n // 2, 0)
            elif kind == "smooth_transition":
                z += 100 / (1 + np.exp(-(t - n // 2) / 5))
            if kind.startswith("step") or kind == "two_steps":
                k = int(rng.integers(n // 3, 2 * n // 3))
                amplitude = {"step_12": 12, "step_30": 30, "step_100": 100}.get(kind, 60)
                z[k:] += amplitude * sigma
                truth = [k]
                if kind == "two_steps":
                    k = n // 3
                    k2 = 2 * n // 3
                    z = 100 + 0.1 * t
                    z[k:] += 60 * sigma
                    z[k2:] += 80 * sigma
                    truth = [k, k2]
            observed = z + rng.normal(0, sigma, n)
            if kind in ("isolated_outliers", "step_with_outliers"):
                idx = rng.choice(n, max(1, n // 15), replace=False)
                observed[idx] += rng.choice([-1, 1], len(idx)) * rng.uniform(30, 300, len(idx)) * sigma
            elif kind.startswith("burst_"):
                length = int(kind.split("_")[1])
                k = int(rng.integers(35, n - length - 30))
                observed[k : k + length] += rng.choice([-1, 1]) * 100 * sigma
            elif kind == "endpoint_burst":
                length = int(rng.integers(3, 20))
                observed[:length] += 100 * sigma
                observed[-length:] -= 80 * sigma
            elif kind == "heteroscedastic":
                observed = z + rng.normal(0, np.linspace(0.5, 10, n) * sigma, n)
                observed[rng.choice(n, 8, replace=False)] += 100 * sigma
            if kind in ("step_gap", "long_gap"):
                k = truth[0] if truth else n // 2
                t[k:] += 9 if kind == "step_gap" else 100
                if kind == "long_gap":
                    observed = 100 + 0.2 * t + rng.normal(0, sigma, n)
            yield dict(kind=kind, rep=rep, t=t, z=observed, truth=truth, sigma=sigma)
    if stress:
        for kind in (
            "long_depth_bias",
            "unresolved_fast_motion",
            "correlated_noise",
            "endpoint_switch",
            "short_swap_return",
        ):
            for rep in range(repeats):
                n = 300
                t = np.arange(n, dtype=float)
                sigma = rng.uniform(0.5, 2)
                z = 100 + 0.1 * t + rng.normal(0, sigma, n)
                truth = []
                if kind == "long_depth_bias":
                    z[100:180] += 100 * sigma
                elif kind == "unresolved_fast_motion":
                    z += 100 * sigma / (1 + np.exp(-(t - 150) / 0.8))
                elif kind == "correlated_noise":
                    residual = rng.normal(0, sigma, n)
                    for i in range(1, n):
                        residual[i] += 0.97 * residual[i - 1]
                    z = 100 + 0.1 * t + residual
                elif kind == "endpoint_switch":
                    z[12:] += 100 * sigma
                    truth = [12]
                else:
                    z[100:132] += 100 * sigma
                    truth = [100, 132]
                yield dict(kind=kind, rep=rep, t=t, z=z, truth=truth, sigma=sigma)


def single_step(t, z):
    s = max(_noise(t, z), 0.5)
    v = np.diff(z) / np.diff(t)
    return list(np.flatnonzero(abs(v - np.median(v)) > 8 * s) + 1)


def global_m(t, z):
    """Recursively split a global quadratic M fit using bounded fit costs."""
    n = len(t)
    s = max(0.5, _noise(t, z))
    if n < 48:
        return []
    sm = median_filter(z, size=5, mode="nearest")
    dt = np.diff(t)
    innovation = abs(np.diff(sm) - np.median(np.diff(sm) / dt) * dt)
    ks = np.flatnonzero(innovation > 4 * s) + 1
    ks = ks[(ks >= 24) & (ks <= n - 24)]
    if not len(ks):
        return []
    tt = (t - t.mean()) / np.ptp(t)
    a = np.vander(tt, 3, increasing=True)
    full = _huber_fit(a, z, s)
    c0 = np.mean(_cost(z - a @ full, s))
    best = None
    for k in ks:
        left = _huber_fit(a[:k], z[:k], s)
        right = _huber_fit(a[k:], z[k:], s)
        gain = c0 - np.mean(_cost(np.r_[z[:k] - a[:k] @ left, z[k:] - a[k:] @ right], s))
        mid = (a[k] + a[k - 1]) / 2
        if abs(mid @ (right - left)) > 8 * s and gain > 0.15 and (best is None or gain > best[0]):
            best = (gain, int(k))
    if best is None:
        return []
    k = best[1]
    return global_m(t[:k], z[:k]) + [k] + [k + j for j in global_m(t[k:], z[k:])]


def run(seed, repeats, output, stress=False):
    output.mkdir(parents=True, exist_ok=True)
    samples = list(cases(seed, repeats, stress))
    methods = {"single_step": single_step, "global_quadratic_m": global_m}
    records = []
    timings = {}
    for name in ("single_step", "global_quadratic_m", "local_without_return", "local_persistent"):
        started = time.perf_counter()
        predictions = []
        if name in methods:
            predictions = [methods[name](c["t"], c["z"]) for c in samples]
        else:
            # One population per acquisition; no true sigma is passed to the method.
            frame = pd.concat([pd.DataFrame({"frame": c["t"], "z_lap": c["z"]}) for c in samples], ignore_index=True)
            labels = np.repeat(np.arange(len(samples)), [len(c["t"]) for c in samples])
            if name == "local_without_return":
                with patch("holod3_linker.depth_repair._returns", return_value=False):
                    result = repair_depth_links(frame, labels)
            else:
                result = repair_depth_links(frame, labels)
            for index, c in enumerate(samples):
                frames = result.cuts.loc[result.cuts.parent_track_id == index, "frame_after"]
                predictions.append([int(np.searchsorted(c["t"], f)) for f in frames])
        timings[name] = time.perf_counter() - started
        for case, cs in zip(samples, predictions, strict=True):
            remaining = list(cs)
            found = 0
            exact = 0
            for k in case["truth"]:
                near = [c for c in remaining if abs(c - k) <= 3]
                if near:
                    match = min(near, key=lambda c: abs(c - k))
                    remaining.remove(match)
                    found += 1
                    exact += int(match == k)
            records.append(
                dict(
                    method=name,
                    kind=case["kind"],
                    rep=case["rep"],
                    true_cuts=len(case["truth"]),
                    recovered=found,
                    exact=exact,
                    extra_cuts=len(remaining),
                    cuts=len(cs),
                    unchanged=int(not cs),
                )
            )
        print(name, round(timings[name], 3), flush=True)
    tab = pd.DataFrame(records)
    summary = tab.groupby(["method", "kind"]).agg(
        tracks=("rep", "count"),
        true_cuts=("true_cuts", "sum"),
        recovered=("recovered", "sum"),
        exact=("exact", "sum"),
        extra_cuts=("extra_cuts", "sum"),
        unchanged=("unchanged", "sum"),
    )
    tab.to_csv(output / "cases.csv", index=False)
    summary.to_csv(output / "summary.csv")
    aggregate = {}
    for name in timings:
        r = tab[tab.method == name]
        positive = r[r.true_cuts > 0]
        negative = r[r.true_cuts == 0]
        aggregate[name] = dict(
            elapsed_sec=timings[name],
            true_cuts=int(positive.true_cuts.sum()),
            recovered_within_3=int(positive.recovered.sum()),
            exact_cuts=int(positive.exact.sum()),
            extra_cuts_positive=int(positive.extra_cuts.sum()),
            intact_controls=int(negative.unchanged.sum()),
            control_tracks=len(negative),
            false_cuts_controls=int(negative.cuts.sum()),
        )
    payload = dict(
        seed=seed,
        repeats=repeats,
        stress=stress,
        tracks=len(samples),
        rows=sum(len(c["t"]) for c in samples),
        methods=aggregate,
    )
    (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=260716)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stress", action="store_true", help="Also expose ambiguous/unresolved-motion failure modes.")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    run(args.seed, args.repeats, args.output, args.stress)
