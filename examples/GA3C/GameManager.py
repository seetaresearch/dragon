# --------------------------------------------------------
# GA3C for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import gym


class GameManager:
    def __init__(self, game_name, display):
        self.game_name = game_name
        self.display = display

        self.env = gym.make(game_name)
        self.reset()

    def reset(self):
        observation = self.env.reset()
        return observation

    def step(self, action):
        self._update_display()
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def _update_display(self):
        if self.display:
            self.env.render()
