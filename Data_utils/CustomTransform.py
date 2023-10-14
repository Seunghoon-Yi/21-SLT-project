import numpy as np
import random

class RandomZeroOut(object):
    def __init__(self, p = 0.15):
        self.p = p

    def __call__(self, clip):
        n_frames = clip.size(0)
        n = int(n_frames * self.p)
        sampled_idx = random.sample(range(0, n_frames), n)
        clip[sampled_idx] = 1.e-5

        return clip