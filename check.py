import copy
import numpy as np
from region import Region, relu
from brain import Brain, LINK

# Checks the TODOs in order and stops at the first one that is missing or wrong.
# Run it after every TODO:  python check.py
#
# The backward checks use "gradient checking": nudge a number a tiny bit up and down,
# see how much the result changes, and compare that with what backward() calculated.


def numeric_grad(f, array, index, eps=1e-6):
    old = array[index]
    array[index] = old + eps
    up = f()
    array[index] = old - eps
    down = f()
    array[index] = old
    return (up - down) / (2 * eps)


def close(a, b):
    return abs(a - b) <= 1e-4 * max(1.0, abs(a), abs(b))


def check_todo_1():
    np.random.seed(1)
    region = Region(3, depth=2, width=4, n_out=5)
    x = np.random.randn(6, 3)
    expected = x
    for W, b in zip(region.weights, region.biases):
        expected = relu(expected @ W + b)
    out = region.forward(x)
    if out.shape != (6, 5):
        return f"forward() should give shape (6, 5), got {out.shape}"
    if not np.allclose(out, expected):
        return "forward() gives the wrong numbers. Check the ReLU line."


def region_loss(region, x, g):
    # a made-up "loss" so we can measure gradients: sum of output × fixed random numbers
    return np.sum(region.forward(x) * g)


def check_region_backward(residual, output):
    np.random.seed(2)
    region = Region(3, depth=3, width=4, n_out=2, residual=residual, output=output)
    x = np.random.randn(5, 3)
    g = np.random.randn(5, 2)

    frozen = copy.deepcopy(region)
    frozen.forward(x)
    grad_in = frozen.backward(g, lr=0.0)      # lr 0 → weights don't move, we only read the gradient
    if grad_in is None or np.shape(grad_in) != x.shape:
        return f"backward() should return the input gradient with shape {x.shape}, got {np.shape(grad_in)}"
    for index in [(0, 0), (2, 1), (4, 2)]:
        expect = numeric_grad(lambda: region_loss(region, x, g), x, index)
        if not close(grad_in[index], expect):
            return f"input gradient is wrong (got {grad_in[index]:.5f}, expected {expect:.5f}). Check TODO 2 and TODO 3."

    moved = copy.deepcopy(region)
    moved.forward(x)
    moved.backward(g, lr=1.0)
    for layer in range(region.n_layers):
        step = region.weights[layer] - moved.weights[layer]   # = gradient, since lr = 1
        expect = numeric_grad(lambda: region_loss(region, x, g) / len(x), region.weights[layer], (0, 0))
        if not close(step[0, 0], expect):
            return f"weight gradient in layer {layer} is wrong (got {step[0, 0]:.5f}, expected {expect:.5f})."


def check_todo_2_and_3():
    for residual, output in [(False, "relu"), (False, "sigmoid"), (True, "relu")]:
        problem = check_region_backward(residual, output)
        if problem:
            return f"{problem}  [region with residual={residual}, output={output}]"


def check_todo_4():
    np.random.seed(3)
    brain = Brain(n_inputs=2, n_actions=1)
    x = np.random.randn(4, 2)
    last = np.random.rand(4, 1)
    out = brain.think(x, last)
    if out is None or np.shape(out) != (4, 1):
        return f"think() should return the answer with shape (4, 1), got {np.shape(out)}"
    p = brain.perception.forward(x)
    f = brain.feedback.forward(last)
    expected = brain.motor.forward(brain.prefrontal.forward(np.concatenate([p, f], axis=1)))
    if not np.allclose(out, expected):
        return "think() gives different numbers than the Brain Map. Check the order and the merge."


def check_todo_5():
    np.random.seed(4)
    brain = Brain(n_inputs=2, n_actions=1)
    x = np.random.randn(5, 2)
    last = np.random.rand(5, 1)
    g = np.random.randn(5, 1)

    def loss():
        return np.sum(brain.think(x, last) * g) / len(x)

    moved = copy.deepcopy(brain)
    moved.think(x, last)
    moved.learn(g, lr=1.0)
    for name in ["motor", "prefrontal", "perception", "feedback"]:
        before = brain.regions()[name].weights[0]
        after = moved.regions()[name].weights[0]
        if np.allclose(before, after):
            return f"learn() never updated {name}. Every region must get its backward() call."
        expect = numeric_grad(loss, before, (0, 0))
        if not close(before[0, 0] - after[0, 0], expect):
            return (f"{name} learned with the wrong gradient (got {before[0, 0] - after[0, 0]:.5f}, "
                    f"expected {expect:.5f}). Check the order and the split at LINK.")


checks = [
    ("TODO 1  region.forward  (ReLU layer)", check_todo_1),
    ("TODO 2+3 region.backward (ReLU slope, pass error back)", check_todo_2_and_3),
    ("TODO 4  brain.think     (wire the regions)", check_todo_4),
    ("TODO 5  brain.learn     (wiring backwards + split)", check_todo_5),
]

for name, check in checks:
    try:
        problem = check()
    except NotImplementedError as e:
        print(f"[ ]  {name}: not done yet ({e})")
        break
    except Exception as e:
        print(f"[X]  {name}: crashed: {type(e).__name__}: {e}")
        break
    if problem:
        print(f"[X]  {name}: {problem}")
        break
    print(f"[OK] {name}")
else:
    print("\nAll TODOs done. Now run:  python train.py")
