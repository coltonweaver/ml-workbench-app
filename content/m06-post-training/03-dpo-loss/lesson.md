---
id: "6.3"
title: Learning from preferences
concepts: [dpo, preference-learning, implicit-reward]
file: m06/dpo.py
stub: stubs/dpo.py
tests: test_dpo_loss.py
predict:
  kind: number
  prompt: >
    At the start of DPO training the policy equals the reference model, so
    every implicit reward is 0. What is the DPO loss (in nats) at that
    point? Give two decimals.
  answer: 0.69
  tolerance: 0.01
mutants:
  - name: forgets the reference
    file: m06/dpo.py
    replace: ["    chosen = beta * (policy_chosen - ref_chosen)\n    rejected = beta * (policy_rejected - ref_rejected)", "    chosen = beta * policy_chosen\n    rejected = beta * policy_rejected"]
  - name: prefers the rejected one
    file: m06/dpo.py
    replace: ["return -F.logsigmoid(chosen - rejected).mean()", "return -F.logsigmoid(rejected - chosen).mean()"]
  - name: counts the prompt
    file: m06/dpo.py
    replace: ["return (logp * mask).sum(-1)", "return logp.sum(-1)"]
warmups:
  - kind: number
    prompt: "At the start of DPO the policy equals the reference. What is the DPO loss? (to 3 decimals)"
    answer: 0.693
    tolerance: 0.02
    explain: "Both log-ratios are 0, so the loss is −log σ(0) = ln 2, whatever β is."
---

## Hook

People can't easily write the perfect response, but they can say which of
two is better. RLHF (Ouyang et al., 2022) learns a reward model from those
comparisons and then optimizes the policy against it with PPO, a fiddly
pipeline with four models in memory. Rafailov et al., "Direct Preference
Optimization" (2023), showed that the same objective has a closed-form
solution that turns preference pairs straight into a classification loss
on the policy.

## Build

In `m06/dpo.py`:

1. `sequence_logprob(model, x, y, mask)`: the summed log-probability of
   each row's response tokens (the mask from 6.1 picks them).
2. `dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected,
   beta)`: with implicit reward r = β·(log π − log π_ref) for each response,
   the loss is −log σ(r_chosen − r_rejected). Return the loss and both
   rewards (detached, for logging).

```python
lp = torch.tensor([-3.0])
dpo_loss(lp, lp - 1, lp, lp - 1, beta=0.1)     # policy == reference
```

## Explain

When the policy equals the reference, both rewards are 0, so the loss is
−log σ(0) = log 2 ≈ **0.69**. Training moves the loss down by making the
chosen response more likely than the reference finds it, and the rejected
one less likely.

The derivation: the RLHF objective (maximize reward minus β·KL to the
reference) is optimized by π*(y) ∝ π_ref(y)·exp(r(y)/β). Invert that and
the reward is β·log(π*/π_ref) plus a term that depends only on the prompt.
Put that into the Bradley–Terry preference model, P(chosen ≻ rejected) =
σ(r_c − r_r), and the prompt term cancels. What's left is a loss on the
policy alone: the policy is its own reward model.

β sets how far the policy may drift from the reference. The reference
term is not optional (the first mutant drops it): without it, DPO just
pushes chosen up and rejected down without bound, and the model degrades.

## Stretch

Compute the gradient of the DPO loss with respect to the policy's log π of
the chosen response by hand. Show it equals β·σ(r_r − r_c): pairs the model
already ranks correctly contribute little, mistakes contribute a lot.
