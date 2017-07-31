# --------------------------------------------------------
# GA3C for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import sys

if sys.version_info >= (3, 0):
    from queue import Queue as queueQueue
else:
    from Queue import Queue as queueQueue

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time

from Config import Config


class ProcessStats(Process):
    def __init__(self):
        super(ProcessStats, self).__init__()
        self.episode_log_q = Queue(maxsize=100)
        self.episode_count = Value('i', 0)
        self.training_count = Value('i', 0)
        self.should_save_model = Value('i', 0)
        self.trainer_count = Value('i', 0)
        self.predictor_count = Value('i', 0)
        self.agent_count = Value('i', 0)
        self.total_frame_count = 0

    def FPS(self):
        # average FPS from the beginning of the training (not current FPS)
        return np.ceil(self.total_frame_count / (time.time() - self.start_time))

    def TPS(self):
        # average TPS from the beginning of the training (not current TPS)
        return np.ceil(self.training_count.value / (time.time() - self.start_time))

    def run(self):
        with open(Config.RESULTS_FILENAME, 'a') as results_logger:
            rolling_frame_count = 0
            rolling_reward = 0
            results_q = queueQueue(maxsize=Config.STAT_ROLLING_MEAN_WINDOW)

            self.start_time = time.time()
            first_time = datetime.now()
            while True:
                episode_time, reward, length = self.episode_log_q.get()
                results_logger.write('%s, %d, %d\n' % (episode_time.strftime("%Y-%m-%d %H:%M:%S"), reward, length))
                results_logger.flush()

                self.total_frame_count += length
                self.episode_count.value += 1

                rolling_frame_count += length
                rolling_reward += reward

                if results_q.full():
                    old_episode_time, old_reward, old_length = results_q.get()
                    rolling_frame_count -= old_length
                    rolling_reward -= old_reward
                    first_time = old_episode_time

                results_q.put((episode_time, reward, length))

                if self.episode_count.value % Config.SAVE_FREQUENCY == 0:
                    self.should_save_model.value = 1

                if self.episode_count.value % Config.PRINT_STATS_FREQUENCY == 0:
                    print(
                        '[Time: %8d] '
                        '[Episode: %8d Score: %10.4f] '
                        '[RScore: %10.4f RPPS: %5d] '
                        '[PPS: %5d TPS: %5d] '
                        '[NT: %2d NP: %2d NA: %2d]'
                        % (int(time.time() - self.start_time),
                           self.episode_count.value, reward,
                           rolling_reward / results_q.qsize(),
                           rolling_frame_count / (datetime.now() - first_time).total_seconds(),
                           self.FPS(), self.TPS(),
                           self.trainer_count.value, self.predictor_count.value, self.agent_count.value))
                    sys.stdout.flush()
