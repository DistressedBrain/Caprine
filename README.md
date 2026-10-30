# Caprine Ai
Inspired by Cyberpunk Edgerunners Season 2.


## Brain Map

```text
          INPUT (text from your "library reader")
                         │
          ┌──────────────┴──────────────┐
   Occipital 16×20               Auditory 10×20 (later)
     (perception)                 (2nd input)
          │                             │
   Temporal 10×32  ─────────────────────┤
   (comprehension)                      │
          └──────────────┬──────────────┘
                 Parietal 6×32  ◄──── Somatosensory 6×20
                 (integration)          (feedback loop)
                         │                     ▲
        ┌────────────────┼────────────────┐    │
 Working memory 8×32     │     Orbitofrontal 12×24
  (holds context) ──► Prefrontal 16×32 ◄── (judges value)
                   (DECIDES, residual)
                         │
                  Cingulate 8×24  (checks for errors → can send back)
                         │
          ┌──────────────┴──────────────┐
    Broca 10×24                    Motor 8×20
   (writes answer)               (takes action)
                                       │
                                       └──► result → Somatosensory
```


## Cortex Map

Each brain region of the agent is a network shape (layers × neurons per layer) that passed the
two-spirals test with ≤ 1.4% average error. Shapes are starting points: each region still has to be
re-tested and trained on its real job.

| Cortex area | What it does in the brain | Job in the AI agent | Network | Why this shape |
|---|---|---|---|---|
| **Occipital lobe** (visual cortex) | Processes vision in stages: edges → shapes → objects | **Perception**: first processing of raw input | **16 × 20** | The visual pathway is a deep hierarchy. Most reliable network (worst run 99.7%), and errors here spoil everything downstream. |
| **Temporal lobe** (Wernicke's area) | Understands language, recognizes objects | **Comprehension**: turns text into meaning | **10 × 32** | Needs width for many concepts and depth for meaning. Top accuracy (99.7%). |
| **Auditory cortex** (temporal) | Processes sound | **Second input channel**, for later (sound, other data) | **10 × 20** | Mid-size, ready when another sense is added |
| **Parietal lobe** (association areas) | Combines senses, spatial reasoning, attention | **Integration**: merges all inputs into one picture | **6 × 32** | Wide takes in many signals at once; shallow merges them quickly. 99.8% accuracy. |
| **Somatosensory cortex** (parietal) | Feels body feedback (touch, position) | **Feedback**: "what happened after my action?" | **6 × 20** | Simple, constant job, so a small fast network |
| **Prefrontal cortex** (dorsolateral) | Planning, reasoning, decisions | **Decision-maker**: the "thinking" region | **16 × 32** + residual | The hardest job gets the most capacity. Must use residual connections to stay stable. |
| **Working memory** (prefrontal) | Keeps a few things "in mind" right now | **Short-term memory**: current goal and recent context | **8 × 32** | Wide holds many items at once. Reliable (worst run 99.0%). |
| **Orbitofrontal cortex** | Judges value: "is this good or bad for me?" | **Value judge**: scores the options before deciding | **12 × 24** | Judgment benefits from depth. Reliable (worst run 98.7%). |
| **Anterior cingulate cortex** | Detects errors and conflicts | **Self-checker**: "did I get that wrong?" | **8 × 24** | Medium job, solid reliability |
| **Broca's area** (frontal) | Produces speech and language | **Language output**: writes answers | **10 × 24** | Balanced all-rounder: good accuracy, fast (2.9 s to 95%) |
| **Motor cortex** (frontal) | Turns decisions into movement | **Action**: executes the chosen action | **8 × 20** | Fastest network (2.5 s to 95%). Actions need quick responses. |

Connections between regions should all be the same width (about 32 neurons) so no link becomes a bottleneck.


## Code

Starter brain: 4 regions (Perception → Prefrontal → Motor, plus Somatosensory feedback), written with NumPy only.

| File | What it is |
|---|---|
| `region.py` | One `Region` class. Every region uses it, only the shape changes. |
| `brain.py` | Creates the regions and wires them together: the Brain Map in code. |
| `train.py` | Trains the brain on the two-spirals practice task (generated in code, nothing downloaded). |
| `check.py` | Checks the TODOs one by one and says what's wrong. |

How it works:
- **Connect:** one region's output is the next region's input.
- **Merge:** `np.concatenate` glues outputs side by side (32 + 32 = 64).
- **Learn:** run the same wiring backwards. `backward()` returns the error for its input, which becomes the error for the region before it. Where outputs were merged, split the error again.
- **Loops:** go through time, not inside one pass. Feedback gets the result of the *previous* action.

Getting started:

```
python check.py     # fill in the TODO in the file it names, run again, repeat
python train.py     # once every check says OK
```

Expected result after training: about 97–99% test accuracy (3,000 steps, about 1 minute).