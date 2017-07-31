# --------------------------------------------------------
# GA3C for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from threading import Thread
import numpy as np

from Config import Config


class ThreadPredictor(Thread):
    def __init__(self, server, id):
        super(ThreadPredictor, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        ids = np.zeros(Config.PREDICTION_BATCH_SIZE, dtype=np.uint16)
        states = np.zeros(
            (Config.PREDICTION_BATCH_SIZE, Config.STACKED_FRAMES, Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, ),
            dtype=np.float32)

        while not self.exit_flag:
            ids[0], states[0] = self.server.prediction_q.get()

            size = 1
            while size < Config.PREDICTION_BATCH_SIZE and not self.server.prediction_q.empty():
                ids[size], states[size] = self.server.prediction_q.get()
                size += 1

            batch = states[:size]
            p, v = self.server.model.predict_p_and_v(batch)

            for i in range(size):
                if ids[i] < len(self.server.agents):
                    self.server.agents[ids[i]].wait_q.put((p[i], v[i]))
