import numpy as np
from region import Region

LINK = 32   # every connection between regions is 32 numbers wide


class Brain:
    """The Brain Map from the README, written as code (starter version: 4 regions).

        INPUT ──► Perception ──┐
                               ├──► Prefrontal ──► Motor ──► result
    last result ──► Feedback ──┘                               │
          ▲                                                    │
          └────────────────── next step ◄──────────────────────┘

    To add a region later: create it in __init__, add it to regions(), think() and learn().
    region.py never has to change.
    """

    def __init__(self, n_inputs, n_actions):
        self.perception = Region(n_inputs, depth=16, width=20)                    # Occipital 16×20
        self.feedback = Region(n_actions, depth=6, width=20)                      # Somatosensory 6×20
        self.prefrontal = Region(LINK + LINK, depth=16, width=32, residual=True)  # Prefrontal 16×32
        self.motor = Region(LINK, depth=8, width=20, n_out=n_actions, output="sigmoid")  # Motor 8×20

    def regions(self):
        return {"perception": self.perception, "feedback": self.feedback,
                "prefrontal": self.prefrontal, "motor": self.motor}

    def think(self, x, last_result):
        # TODO 4: run the signal through the regions, following the map above.
        #   1. perception reads x, feedback reads last_result (each gives 32 numbers)
        #   2. merge them: np.concatenate([a, b], axis=1) glues them side by side → 64 numbers
        #   3. prefrontal reads the 64, motor reads prefrontal's 32
        #   4. return motor's output (the answer)
        raise NotImplementedError("TODO 4 in brain.py")

    def learn(self, grad, lr):
        # TODO 5: the same wiring, run backwards. Each region.backward(...) returns the
        #         error for its input, which becomes the grad for the region before it.
        #   1. motor first, then prefrontal
        #   2. prefrontal's returned grad has 64 columns: the first LINK belong to perception,
        #      the rest to feedback. Split it: g[:, :LINK] and g[:, LINK:]
        #   3. give each part to its region
        raise NotImplementedError("TODO 5 in brain.py")
