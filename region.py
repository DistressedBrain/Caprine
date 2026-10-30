import math
import numpy as np


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Region:
    """One brain region: a small network that reads `n_in` numbers and outputs `n_out` numbers.

    depth    = how many hidden layers
    width    = neurons per hidden layer
    residual = add skip connections around hidden layers (needed for deep regions)
    output   = "relu" for regions that feed other regions, "sigmoid" for the very last one
    """

    def __init__(self, n_in, depth, width, n_out=32, residual=False, output="relu"):
        self.residual = residual
        self.output = output
        sizes = [n_in] + [width] * depth + [n_out]
        self.n_layers = len(sizes) - 1
        self.weights = []
        self.biases = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            scale = math.sqrt(2 / a)
            if residual and self.is_skip_layer(i):
                scale /= math.sqrt(depth)
            self.weights.append(np.random.randn(a, b) * scale)
            self.biases.append(np.zeros((1, b)))

    def is_skip_layer(self, i):
        # skip connections only fit where input and output have the same size:
        # the hidden->hidden layers (not the first layer, not the last layer)
        return 0 < i < self.n_layers - 1

    def parameter_count(self):
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    def forward(self, x):
        self.acts = [x]
        last = self.n_layers - 1
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = x @ W + b
            if i == last and self.output == "sigmoid":
                x = sigmoid(z)
            elif self.residual and self.is_skip_layer(i):
                x = x + relu(z)
            else:
                # TODO 1: a normal layer. Turn z into this layer's output with the ReLU activation.
                raise NotImplementedError("TODO 1 in region.py")
            self.acts.append(x)
        return x

    def backward(self, grad_out, lr):
        """grad_out = how much each output number should change (comes from the next region).
        Returns how much each INPUT number should change, so the previous region can learn too."""
        last = self.n_layers - 1
        for i in reversed(range(self.n_layers)):
            x_in, x_out = self.acts[i], self.acts[i + 1]
            if i == last and self.output == "sigmoid":
                dz = grad_out * x_out * (1 - x_out)
            elif self.residual and self.is_skip_layer(i):
                dz = grad_out * (x_out - x_in > 0)
            else:
                # TODO 2: slope of ReLU. The error only passes through neurons that were
                #         active (output > 0). Hint: (x_out > 0) is 1 for active, 0 for off.
                raise NotImplementedError("TODO 2 in region.py")

            # TODO 3: the error for this layer's input. Send dz back through the weights
            #         (the same weights used going forward, but transposed: .T).
            #         This is the line that lets a region pass its error to the region before it.
            raise NotImplementedError("TODO 3 in region.py")

            if self.residual and self.is_skip_layer(i):
                grad_in = grad_in + grad_out      # the skip "highway" passes the error straight through
            self.weights[i] -= lr * (x_in.T @ dz / len(dz))
            self.biases[i] -= lr * dz.mean(axis=0, keepdims=True)
            grad_out = grad_in
        return grad_out
