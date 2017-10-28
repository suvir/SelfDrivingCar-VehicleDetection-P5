import numpy as np


class FrameManager:
    def __init__(self, max_frames):
        self.frames = []
        self.max_frames = max_frames

    def pushFrame(self, frame):
        self.frames.insert(0, frame)

    def _size(self):
        return len(self.frames)

    def _popFrame(self):
        prev_n = len(self.frames)
        self.frames.pop()
        after_n = len(self.frames)
        assert prev_n == (after_n + 1)

    def sum_frames(self):
        if self._size() > self.max_frames:
            self._popFrame()
        all_frames = np.array(self.frames)
        return np.sum(all_frames, axis=0)
