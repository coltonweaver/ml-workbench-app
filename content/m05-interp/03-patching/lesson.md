---
id: "5.3"
title: Activation patching
concepts: [activation-patching, causal-intervention, logit-difference]
file: m05/patching.py
stub: stubs/patching.py
tests: test_patching.py
predict:
  kind: choice
  prompt: >
    Two prompts differ only in the token at position 2. You run the
    corrupted prompt, but replace the residual stream entering block 0 at
    position 0 with its value from the clean prompt. What fraction of the
    clean behavior comes back?
  options:
    - all of it
    - none of it
    - about half
    - it depends on the weights
  answer: 1
mutants:
  - name: patches every position
    file: m05/patching.py
    replace: ["        t[:, pos] = source[name][:, pos]", "        t[:] = source[name]"]
  - name: normalizes the wrong way round
    file: m05/patching.py
    replace: ["(float(metric(logits)) - m_corrupt) / (m_clean - m_corrupt)", "(float(metric(logits)) - m_clean) / (m_corrupt - m_clean)"]
  - name: logit difference backwards
    file: m05/patching.py
    replace: ["return (last[:, good] - last[:, bad]).mean()", "return (last[:, bad] - last[:, good]).mean()"]
---

## Hook

Attribution reads the network; patching experiments on it. Take two inputs
that differ in one detail (the **clean** and **corrupted** prompts), so the
model's answer differs too. Run the corrupted one, but swap in a single
activation from the clean run. If the clean answer comes back, that
activation carries the information that made the difference.

Meng et al., "Locating and Editing Factual Associations in GPT" (2022),
used this to find where GPT stores facts; Wang et al. (2022) used it to map
the whole circuit GPT-2 uses for indirect object identification.

## Build

In `m05/patching.py`:

1. `logit_diff(logits, good, bad, pos=-1)`: logit[good] − logit[bad], a
   good metric because it ignores everything the two answers share.
2. `patch_position(name, pos, source)`: a patch function for `run` that
   replaces one position with the value from a clean cache. Clone first.
3. `patching_sweep(model, clean, corrupt, metric, names)`: for every name
   and position, the normalized effect
   (patched − corrupt) / (clean − corrupt).

```python
from workbench import show_matrix
clean = encode("my lord, I")
corrupt = encode("my lady, I")     # same length; pick a pair your model answers differently
good, bad = stoi["a"], stoi["w"]   # and two next characters that tell them apart
names = [f"blocks.{i}.resid_pre" for i in range(len(m.blocks))] + [f"blocks.{len(m.blocks) - 1}.resid_post"]
sweep = patching_sweep(m, clean[None], corrupt[None], lambda l: logit_diff(l, good, bad), names)
show_matrix(sweep, row_labels=names, col_labels=list(decode(clean)), title="patching")
```

## Explain

Position 0 is identical in both prompts, so its stream entering block 0 is
identical too (it's just the embedding), and swapping it changes
**none of it**. In the first layer only the changed token's position
carries the difference; layer by layer, attention moves it toward the
final position. A patching sweep over layers and positions draws that
movement.

Normalizing makes sweeps comparable: 1 means this one activation is
sufficient to restore the clean behavior, 0 that it doesn't matter, and
values outside [0, 1] mean something interesting (patching overshoots, or
pushes the wrong way).

This is the **denoising** direction (corrupted run, clean patch): it finds
what is sufficient. The reverse, **noising** (clean run, corrupted patch),
finds what is necessary. They can disagree when a circuit has backups;
Heimersheim and Nanda, "How to use and interpret activation patching"
(2024), is the practical guide.

## Stretch

Patch `blocks.{i}.attn.z` one head at a time (patch only `[:, pos, head]`)
at the final position. Which heads matter? Then run the noising version and
compare.
