import numpy as np

class Memory():
    def __init__(self, max_size=100000, fast_sample_threshold=2000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size=max_size
        self.fast_sample_threshold = fast_sample_threshold
        self.rng = np.random.default_rng()

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx,:] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        if self.size == 0 or batch == 0:
            return self.transitions[0:0]

        # For large buffers, sampling without replacement becomes expensive.
        # Use fast sampling with replacement to keep update time stable.
        if self.size >= self.fast_sample_threshold and batch < self.size:
            inds = self.rng.integers(0, self.size, size=batch)
        else:
            inds = self.rng.choice(self.size, size=batch, replace=False)
        return self.transitions[inds,:]

    def get_all_transitions(self):
        return self.transitions[0:self.size]
