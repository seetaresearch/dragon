# --------------------------------------------------------
# GA3C for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
import scipy.misc as misc

from Config import Config
from GameManager import GameManager

class Queue(object):
    def __init__(self, maxsize):
        self._maxsize = maxsize
        self.queue = []

    def full(self):
        return len(self.queue) >= self._maxsize

    def get(self):
        ret = self.queue[0]
        self.queue = self.queue[1:]
        return ret

    def put(self, x):
        self.queue.append(x)

    def clear(self):
        self.queue = []


class Environment:
    def __init__(self):
        self.game = GameManager(Config.ATARI_GAME, display=Config.PLAY_MODE)
        self.nb_frames = Config.STACKED_FRAMES
        self.frame_q = Queue(maxsize=self.nb_frames)
        self.previous_state = None
        self.current_state = None
        self.total_reward = 0

        self.reset()

    @staticmethod
    def _rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    @staticmethod
    def _preprocess(image):
        image = Environment._rgb2gray(image)
        image = misc.imresize(image, [Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH], 'bilinear')
        image = image.astype(np.float32) / 128.0 - 1.0
        return image

    def _get_current_state(self):
        if not self.frame_q.full():
            return None  # frame queue is not full yet.
        x_ = np.array(self.frame_q.queue, dtype=np.float32)
        return x_

    def _update_frame_q(self, frame):
        if self.frame_q.full():
            self.frame_q.get()
        image = Environment._preprocess(frame)
        self.frame_q.put(image)

    def get_num_actions(self):
        return self.game.env.action_space.n

    def reset(self):
        self.total_reward = 0
        self.frame_q.clear()
        self._update_frame_q(self.game.reset())
        self.previous_state = self.current_state = None

    def step(self, action):
        observation, reward, done, _ = self.game.step(action)

        self.total_reward += reward
        self._update_frame_q(observation)

        self.previous_state = self.current_state
        self.current_state = self._get_current_state()
        return reward, done