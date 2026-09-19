"""Certify the seven-point aliasing witness using rational interval arithmetic.

JSON decimal coordinate strings are interpreted as exact rational numbers.
No numpy, torch, or floating-point arithmetic is used for the certificate.
All anchored directed tours are enumerated independently of the forest code.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path


def decimal_string(integer, scale):
    with localcontext() as ctx:
        ctx.prec = 100
        return str(Decimal(integer) / Decimal(scale))


def verify(source, output, digits=15):
    raw = source.read_bytes()
    # parse_float=str preserves the exact supplied decimal representation.
    case = json.loads(raw, parse_float=str)
    x = [[Fraction(v) for v in point] for point in case['coordinates']]
    n, scale = len(x), 10**digits
    low_dist = [[0] * n for _ in x]
    high_dist = [[0] * n for _ in x]
    for i in range(n):
        for j in range(n):
            q = sum((a - b)**2 for a, b in zip(x[i], x[j]))
            k = math.isqrt(q.numerator * scale**2 // q.denominator)
            exact = k*k*q.denominator == q.numerator*scale**2
            low_dist[i][j] = k
            high_dist[i][j] = k if exact else k+1
            assert k*k*q.denominator <= q.numerator*scale**2
            assert high_dist[i][j]**2*q.denominator >= q.numerator*scale**2
    lower_q, upper_q, valid_counts = [], [], []
    for edges in case['partial_edges']:
        lower, upper, count = {}, {}, 0
        for rest in itertools.permutations(range(1, n)):
            tour = (0,) + rest
            succ = {tour[i]: tour[(i+1) % n] for i in range(n)}
            if not all(succ[s] == t for s, t in edges):
                continue
            count += 1
            action = succ[case['source']]
            lo = sum(low_dist[i][j] for i, j in succ.items())
            hi = sum(high_dist[i][j] for i, j in succ.items())
            lower[action] = min(lower.get(action, lo), lo)
            upper[action] = min(upper.get(action, hi), hi)
        assert count > 0
        lower_q.append(lower)
        upper_q.append(upper)
        valid_counts.append(count)
    actions = sorted(lower_q[0])
    assert all(sorted(q) == actions for q in lower_q + upper_q)
    states = len(lower_q)
    lower = min(sum(q[a] for q in lower_q) for a in actions) - sum(min(q.values()) for q in upper_q)
    upper = min(sum(q[a] for q in upper_q) for a in actions) - sum(min(q.values()) for q in lower_q)
    assert lower > 0
    lo_num = {a: sum(q[a] for q in lower_q) for a in actions}
    hi_num = {a: sum(q[a] for q in upper_q) for a in actions}
    certified_best = [a for a in actions if all(hi_num[a] < lo_num[b] for b in actions if b != a)]
    result = {
        'kind': 'rational interval certificate',
        'coordinate_semantics': 'JSON decimal strings are exact rationals',
        'enumerated_anchored_tours_per_forest': math.factorial(n-1),
        'valid_completion_counts': valid_counts,
        'distance_interval_unit': str(Fraction(1, scale)),
        'legal_heads': actions,
        'source': case['source'],
        'equal_prior': True,
        'certified_optimal_shared_heads': certified_best,
        'bayes_regret_lower': decimal_string(lower, states*scale),
        'bayes_regret_upper': decimal_string(upper, states*scale),
        'bayes_regret_lower_rational': str(Fraction(lower, states*scale)),
        'bayes_regret_upper_rational': str(Fraction(upper, states*scale)),
        'conditional_q_intervals': [
            {str(a): [decimal_string(lo[a], scale), decimal_string(hi[a], scale)] for a in actions}
            for lo, hi in zip(lower_q, upper_q)
        ],
        'input_sha256': hashlib.sha256(raw).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=root/'endpoint_aliasing_witness.json')
    parser.add_argument('--output', type=Path, default=root/'witness_interval_certificate.json')
    parser.add_argument('--digits', type=int, default=15)
    args = parser.parse_args()
    if args.digits < 1:
        parser.error('--digits must be positive')
    verify(args.source, args.output, args.digits)
