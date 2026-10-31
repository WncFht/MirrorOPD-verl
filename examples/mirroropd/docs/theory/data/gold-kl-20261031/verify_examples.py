# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic checks; not model experiments and not a proof substitute."""
import json
import math
from pathlib import Path


def kl(p, q):
    return sum(a * math.log(a / b) for a, b in zip(p, q, strict=True))


def softmax(z):
    v = [math.exp(x - max(z)) for x in z]
    return [x / sum(v) for x in v]


def binary(a):
    return [1 - a, a]


def joint(root, cond):
    return [binary(root)[a] * binary(cond[a])[b] for a in range(2) for b in range(2)]


def main():
    p = [.6, .01, .39]
    rewards = [1, 1, 0]
    gold = sum(a * r for a, r in zip(p, rewards, strict=True))
    grad_gold = [a * (r - gold) for a, r in zip(p, rewards, strict=True)]
    teachers = []
    for label, q in [('A', [.01, .89, .1]), ('B', [.7, .1, .2])]:
        loss = kl(p, q)
        direction = [-a * (math.log(a / b) - loss) for a, b in zip(p, q, strict=True)]
        derivative = sum(a * b for a, b in zip(grad_gold, direction, strict=True))
        def score(eta):
            after = softmax([math.log(a) + eta * b for a, b in zip(p, direction, strict=True)])
            return after[0] + after[1]
        numerical = (score(1e-6) - score(-1e-6)) / 2e-6
        assert abs(numerical - derivative) < 1e-8
        teachers.append(dict(teacher=label, probabilities=q, own_accuracy=q[0]+q[1],
                             student_initial_accuracy=gold, rkl=loss, negative_rkl_logit_gradient=direction,
                             analytic_gold_derivative=derivative, finite_difference=numerical,
                             student_accuracy_after_eta_001=score(.01)))
    assert teachers[0]['analytic_gold_derivative'] < 0 < teachers[1]['analytic_gold_derivative']

    pr, qr, pc, qc = .6, .45, [.3, .9], [.1, .6]
    P, Q = joint(pr, pc), joint(qr, qc)
    seq = kl(P, Q)
    chain = kl(binary(pr), binary(qr)) + (1-pr)*kl(binary(pc[0]), binary(qc[0])) + pr*kl(binary(pc[1]), binary(qc[1]))
    assert abs(seq-chain) < 1e-12
    gp, gq = P[1]+P[3], Q[1]+Q[3]
    assert abs(gp-gq) <= math.sqrt(seq/2)
    correct_kl = kl([P[1]/gp, P[3]/gp], [Q[1]/gq, Q[3]/gq])
    wrong_kl = kl([P[0]/(1-gp), P[2]/(1-gp)], [Q[0]/(1-gq), Q[2]/(1-gq)])
    partition = kl(binary(gp), binary(gq)) + gp*correct_kl + (1-gp)*wrong_kl
    assert abs(seq-partition) < 1e-12

    same_acc = []
    for eps in [1e-3, 1e-6, 1e-12]:
        same_acc.append(dict(epsilon=eps, p_accuracy=.5, q_accuracy=.5, kl=kl([.25,.25,.5],[eps,.5-eps,.5])))
    assert same_acc[0]['kl'] < same_acc[1]['kl'] < same_acc[2]['kl']

    quadratic = []
    for tau in [.001, .0005]:
        prob = [.25+tau, .25-tau-tau*tau, .5+tau*tau]
        k = kl(prob,[.25,.25,.5])
        gain = prob[0]+prob[1]-.5
        assert abs(gain/k+.25) < .001
        quadratic.append(dict(tau=tau, kl=k, accuracy_gain=gain, gain_over_kl=gain/k))

    bs, mu = [.2,.7],[.4,.6]
    variance = sum(w*b*(1-b) for w,b in zip(mu,bs,strict=True))
    envelope = []
    for eta in [.001,.0005]:
        aa = [b*math.exp(eta)/(1-b+b*math.exp(eta)) for b in bs]
        gain = sum(w*(a-b) for w,a,b in zip(mu,aa,bs,strict=True))
        k = sum(w*kl(binary(a),binary(b)) for w,a,b in zip(mu,aa,bs,strict=True))
        slope = gain/math.sqrt(k)
        assert abs(slope-math.sqrt(2*variance)) < .001
        envelope.append(dict(eta=eta, kl=k, gain=gain, gain_over_sqrt_kl=slope,
                             predicted_limit=math.sqrt(2*variance)))
    result = dict(kind='synthetic mathematical examples; no LLM training', teacher_counterexample=teachers,
                  chain_rule=dict(sequence_kl=seq, conditional_sum=chain, fixed_h_token_mean=chain/2,
                                  accuracy_p=gp, accuracy_q=gq, correctness_partition_kl=partition),
                  equal_accuracy_different_kl=same_acc, zero_linear_coefficient=quadratic,
                  optimal_frontier=envelope, checks='all assertions passed')
    Path(__file__).with_name('numerical-checks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Synthetic checks passed: gradient finite differences, chain rule, correctness partition, counterexamples, optimal frontier.')


if __name__ == '__main__':
    main()
