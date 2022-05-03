from typing import Dict, List, Tuple
from collections import deque
import random


class Buffer(object):
    def __init__(self, memory_size) -> None:
        super(Buffer, self).__init__()

        self.size = memory_size
        self.data = deque(maxlen=self.size)
    
    def save_trans(self, transition: Tuple) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.data, batch_size)
        obs, a, r, done, obs_ = zip(*batch)
        obs, a, r, done, obs_ = np.stack(obs, 0), np.stack(a, 0), np.array(r), np.array(done), np.stack(obs_, 0)
        return obs, a, r, done, obs_

    def __len__(self):
        return len(self.data)