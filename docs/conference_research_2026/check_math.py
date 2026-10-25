"""Small deterministic checks for the definitions, not research experiments.

Run from any directory with Python + NumPy + Matplotlib. No datasets, model
training, or external services. Enumeration is an independent oracle for the DP.
"""
from itertools import combinations
from pathlib import Path
import json
import numpy as np


def extrema_dp(costs, bounds, lengths):
    segments, total = costs.shape
    outputs = []
    for sign in (1.0, -1.0):
        previous = np.full(total + 1, np.inf)
        previous[0] = 0.0
        for j in range(segments):
            prefix = np.r_[0.0, np.cumsum(sign * costs[j])]
            current = np.full(total + 1, np.inf)
            best = np.inf
            lo, hi = bounds[j]
            for t in range(total + 1):
                s = t - lengths[j]
                if s >= 0:
                    best = min(best, previous[s] - prefix[s])
                if lo <= t <= hi:
                    current[t] = prefix[t] + best
            previous = current
        outputs.append(sign * previous[total])
    return np.array(outputs)


def enumerate_values(costs, bounds, lengths):
    segments, total = costs.shape
    values = []
    for internal in combinations(range(1, total), segments - 1):
        endpoints = (0,) + internal + (total,)
        if any(not lo <= endpoints[j + 1] <= hi
               or endpoints[j + 1] - endpoints[j] < lengths[j]
               for j, (lo, hi) in enumerate(bounds)):
            continue
        values.append(sum(costs[j, endpoints[j]:endpoints[j + 1]].sum()
                          for j in range(segments)))
    return np.array(values)


def run():
    rng = np.random.default_rng(20261025)
    root = Path(__file__).resolve().parent
    valid = invalid = 0
    max_error = 0.0
    for _ in range(1000):
        total = int(rng.integers(3, 11))
        segments = int(rng.integers(1, min(5, total) + 1))
        bounds = []
        for j in range(segments):
            if j == segments - 1:
                bounds.append((total, total))
            else:
                lo, hi = sorted(rng.integers(1, total, size=2).tolist())
                bounds.append((lo, hi))
        lengths = rng.integers(1, 3, size=segments)
        weights = rng.dirichlet(np.ones(total))
        loss_f = rng.uniform(size=(segments, total)) * weights
        loss_g = rng.uniform(size=(segments, total)) * weights
        costs = loss_f - loss_g
        brute = enumerate_values(costs, bounds, lengths)
        actual = extrema_dp(costs, bounds, lengths)
        if not len(brute):
            assert np.isposinf(actual[0]) and np.isneginf(actual[1])
            invalid += 1
            continue
        expected = np.array([brute.min(), brute.max()])
        err = float(np.max(np.abs(expected - actual)))
        max_error = max(max_error, err)
        assert err < 1e-12
        f = enumerate_values(loss_f, bounds, lengths)
        g = enumerate_values(loss_g, bounds, lengths)
        assert actual[0] >= f.min() - g.max() - 1e-12
        assert actual[1] <= f.max() - g.min() + 1e-12
        valid += 1

    nullspace_checks = 0
    for _ in range(200):
        dim = 6
        rows = int(rng.integers(1, dim))
        matrix = rng.normal(size=(rows, dim))
        z = rng.normal(size=dim)
        x = matrix @ z
        pinv = np.linalg.pinv(matrix)
        center = pinv @ x
        projector = np.eye(dim) - pinv @ matrix
        radius = float(np.linalg.norm(center) + rng.uniform(0.1, 2.0))
        residual_radius = np.sqrt(radius**2 - center @ center)
        task = rng.normal(size=dim)
        direction = projector @ task
        direction /= np.linalg.norm(direction)
        halfwidth = residual_radius * np.linalg.norm(projector @ task)
        for sign in (-1.0, 1.0):
            endpoint = center + sign * residual_radius * direction
            assert np.allclose(matrix @ endpoint, x, atol=1e-10)
            assert np.isclose(np.linalg.norm(endpoint), radius, atol=1e-10)
            assert np.isclose(task @ endpoint, task @ center + sign * halfwidth,
                              atol=1e-10)
        nullspace_checks += 1

    f = np.array([0, 1, 1, 1])
    g = np.array([0, 0, 0, 1])
    differences = []
    for boundary in (1, 2, 3):
        y = np.r_[np.zeros(boundary), np.ones(4 - boundary)]
        differences.append(float(np.mean(f != y) - np.mean(g != y)))
    assert differences == [-0.5, 0.0, 0.5]

    report = {
        "seed": 20261025,
        "dp_vs_exhaustive_cases": 1000,
        "feasible_cases": valid,
        "infeasible_cases": invalid,
        "max_absolute_endpoint_error": max_error,
        "paired_interval_inclusion_checks": valid,
        "nullspace_endpoint_checks": nullspace_checks,
        "four_frame_counterexample": differences,
        "limitations": "Toy numerical checks only; not a novelty check, real-data result, or proof of the population theorem.",
    }
    (root / 'math_checks.json').write_text(json.dumps(report, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    axes[0].plot([1, 2, 3], differences, 'o-', color='#0072B2', lw=2)
    axes[0].axhline(0, color='gray', lw=1, linestyle='--')
    axes[0].set(xlabel='Feasible shared boundary b', ylabel='Risk(f) - Risk(g)',
                xticks=[1, 2, 3], title='Same observed interval, opposite winners')
    axes[0].text(1.02, -0.42, 'f better', color='#0072B2')
    axes[0].text(2.42, 0.36, 'g better', color='#D55E00')
    axes[1].hlines(1, -0.5, 0.5, lw=7, color='#0072B2')
    axes[1].plot([-0.5, 0.5], [1, 1], '|', color='black', ms=15)
    axes[1].axvline(0, color='gray', lw=1, linestyle='--')
    axes[1].set(xlim=(-0.7, 0.7), ylim=(0.5, 1.5), yticks=[],
                xlabel='Possible paired risk difference', title='Sharp certificate: undecidable')
    axes[1].text(0, 0.75, 'No training required for this counterexample',
                 ha='center', fontsize=9)
    out = root / 'figures'
    out.mkdir(exist_ok=True)
    fig.savefig(out / 'boundary_certificate.png', dpi=180)
    fig.savefig(out / 'boundary_certificate.svg')
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
