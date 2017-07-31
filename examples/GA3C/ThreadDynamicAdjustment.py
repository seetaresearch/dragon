# --------------------------------------------------------
# GA3C for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from threading import Thread

import numpy as np
import time

from Config import Config

class ThreadDynamicAdjustment(Thread):
    def __init__(self, server):
        super(ThreadDynamicAdjustment, self).__init__()
        self.setDaemon(True)

        self.server = server
        self.enabled = Config.DYNAMIC_SETTINGS

        self.trainer_count = Config.TRAINERS
        self.predictor_count = Config.PREDICTORS
        self.agent_count = Config.AGENTS

        self.temporal_training_count = 0
        self.exit_flag = False

    def enable_disable_components(self):
        cur_len = len(self.server.trainers)
        if cur_len < self.trainer_count:
            for _ in np.arange(cur_len, self.trainer_count):
                self.server.add_trainer()
        elif cur_len > self.trainer_count:
            for _ in np.arange(self.trainer_count, cur_len):
                self.server.remove_trainer()

        cur_len = len(self.server.predictors)
        if cur_len < self.predictor_count:
            for _ in np.arange(cur_len, self.predictor_count):
                self.server.add_predictor()
        elif cur_len > self.predictor_count:
            for _ in np.arange(self.predictor_count, cur_len):
                self.server.remove_predictor()

        cur_len = len(self.server.agents)
        if cur_len < self.agent_count:
            for _ in np.arange(cur_len, self.agent_count):
                self.server.add_agent()
        elif cur_len > self.agent_count:
            for _ in np.arange(self.agent_count, cur_len):
                self.server.remove_agent()

    def random_walk(self):
        # 3 directions, 1 for Trainers, 1 for Predictors and 1 for Agents
        # 3 outcome for each, -1: add one, 0: no change, 1: remove one
        direction = np.random.randint(3, size=3) - 1
        self.trainer_count = max(1, self.trainer_count - direction[0])
        self.predictor_count = max(1, self.predictor_count - direction[1])
        self.agent_count = max(1, self.agent_count - direction[2])

    def update_stats(self):
        self.server.stats.trainer_count.value = self.trainer_count
        self.server.stats.predictor_count.value = self.predictor_count
        self.server.stats.agent_count.value = self.agent_count

    def run(self):
        self.enable_disable_components()
        self.update_stats()
        if not self.enabled: return

        # Wait for initialization
        time.sleep(Config.DYNAMIC_SETTINGS_INITIAL_WAIT)

        while not self.exit_flag:
            old_trainer_count, old_predictor_count, old_agent_count = \
                self.trainer_count, self.predictor_count, self.agent_count
            self.random_walk()

            # If no change, do nothing
            if self.trainer_count == old_trainer_count \
                    and self.predictor_count == old_predictor_count \
                    and self.agent_count == old_agent_count:
                continue

            old_count = self.temporal_training_count
            self.enable_disable_components()

            self.temporal_training_count = 0
            time.sleep(Config.DYNAMIC_SETTINGS_STEP_WAIT)

            cur_count = self.temporal_training_count
            # if it didn't work, revert the changes
            if cur_count < old_count:
                self.trainer_count, self.predictor_count, self.agent_count = \
                    old_trainer_count, old_predictor_count, old_agent_count

            self.update_stats()