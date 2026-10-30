import math
import time
import numpy as np
from brain import Brain

# Practice task: the two spirals from our experiments, made right here in code
# (nothing is downloaded). The brain sees a point and must say which spiral it is on.
# Run check.py first: this only works once all TODOs are done.


def make_spirals(points_per_arm, turns, noise, rng):
    t = np.sqrt(rng.random(points_per_arm)) * turns * 2 * math.pi
    arm0 = np.column_stack([t * np.cos(t), t * np.sin(t)])
    arm1 = -arm0
    X = np.vstack([arm0, arm1]) / (turns * 2 * math.pi)
    X += rng.normal(0, noise, X.shape)
    y = np.vstack([np.zeros((points_per_arm, 1)), np.ones((points_per_arm, 1))])
    order = rng.permutation(len(X))
    return X[order], y[order]


STEPS = 3000
LEARNING_RATE = 0.1   # 0.2 was too jumpy: all regions chained together are ~43 layers deep
SEED = 0              # change it to see how much luck matters

rng = np.random.default_rng(42)
X, y = make_spirals(500, turns=2.5, noise=0.02, rng=rng)
X_train, y_train, X_test, y_test = X[:700], y[:700], X[700:], y[700:]

np.random.seed(SEED)
brain = Brain(n_inputs=2, n_actions=1)
for name, region in brain.regions().items():
    print(f"{name:>10}: {region.parameter_count():,} parameters")

last_train = np.full_like(y_train, 0.5)   # no answer yet → "don't know" (0.5)
last_test = np.full_like(y_test, 0.5)
start = time.perf_counter()
for step in range(1, STEPS + 1):
    answer = brain.think(X_train, last_train)
    brain.learn(answer - y_train, LEARNING_RATE)
    # Feedback = the result of the PREVIOUS action (the point before this one).
    # Feeding back the answer for the same point would let the brain cheat by copying it.
    last_train = np.roll(answer, 1, axis=0)
    if step % 250 == 0:
        test_answer = brain.think(X_test, last_test)
        last_test = np.roll(test_answer, 1, axis=0)
        accuracy = np.mean((test_answer > 0.5) == y_test)
        loss = np.mean((answer - y_train) ** 2)
        print(f"step {step:>5}  loss {loss:.4f}  test accuracy {accuracy:.1%}  ({time.perf_counter() - start:.0f} s)")
