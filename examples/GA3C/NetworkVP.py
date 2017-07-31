# --------------------------------------------------------
# GA3C for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import threading
from dragon.core.tensor import Tensor
import dragon.vm.theano as theano
import dragon.vm.theano.tensor as T
import dragon.ops as ops
import dragon.core.workspace as ws
import dragon.updaters as updaters

from Config import Config

mutex = threading.Lock()

class NetworkVP:
    def __init__(self, model_name, num_actions):
        self.model_name = model_name
        self.num_actions = num_actions
        self.network_params = []

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self._create_graph()
        if Config.PLAY_MODE:
            ws

    def _create_graph(self):
        self.x = Tensor(shape=[None, self.img_channels, self.img_height, self.img_width]).Variable()
        self.y_r = Tensor(shape=[None], name='Yr').Variable()

        # As implemented in A3C paper
        self.n1 = ops.Relu(ops.Conv2D([self.x] + self.weight_bias(), kernel_size=8, stride=4, num_output=16))
        self.n2 = ops.Relu(ops.Conv2D([self.n1] + self.weight_bias(), kernel_size=4, stride=2, num_output=32))

        self.action_index = Tensor(shape=[None, self.num_actions]).Variable()

        self.d1 = ops.Relu(ops.InnerProduct([self.n2] + self.weight_bias(), num_output=256))

        self.logits_v = ops.InnerProduct([self.d1] + self.weight_bias(), num_output=1)
        self.cost_v = ops.L2Loss([self.y_r, self.logits_v])

        self.logits_p = ops.InnerProduct([self.d1] + self.weight_bias(), num_output=self.num_actions)

        if Config.USE_LOG_SOFTMAX: raise NotImplementedError()
        else:
            self.softmax_p = ops.Softmax(self.logits_p)
            self.selected_action_prob = ops.Sum(self.softmax_p * self.action_index, axis=1)
            self.cost_p_1 = ops.Log(ops.Clip(self.selected_action_prob, self.log_epsilon, None)) * \
                            (self.y_r - ops.StopGradient(self.logits_v))
            self.cost_p_2 = ops.Sum(ops.Log(ops.Clip(self.softmax_p, self.log_epsilon, None)) *
                                      self.softmax_p, axis=1) * (-self.beta)
        self.cost_p_1_agg = ops.Sum(self.cost_p_1)
        self.cost_p_2_agg = ops.Sum(self.cost_p_2)
        self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)
        self.cost_all = self.cost_p + self.cost_v
        
        if Config.DUAL_RMSPROP: raise NotImplementedError()
        else:
            if Config.USE_GRAD_CLIP:
                self.opt = updaters.RMSPropUpdater(decay=Config.RMSPROP_DECAY,
                                                   eps=Config.RMSPROP_EPSILON,
                                                   clip_gradient=Config.GRAD_CLIP_NORM)
            else:
                self.opt = updaters.RMSPropUpdater(decay=Config.RMSPROP_DECAY,
                                                   eps=Config.RMSPROP_EPSILON)

        grads = T.grad(self.cost_all, self.network_params)
        for p, g in zip(self.network_params, grads):
            self.opt.append((p, g), lr_mult=1.0)

    def weight_bias(self, weights_init=None, no_bias=False):
        if weights_init is None:
            weight = Tensor().Xavier()
        else:
            weight = weights_init
        if no_bias:
            self.network_params.extend([weight])
            return [weight]
        bias = Tensor().Constant(value=0)
        self.network_params.extend([weight, bias])
        return [weight, bias]

    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    def predict_v(self, x):
        if not hasattr(self, '_predict_p'):
            self._predict_v = theano.function(inputs=self.x, outputs=self.logits_v)
        return self._predict_v(x)

    def predict_p(self, x):
        if not hasattr(self, '_predict_p'):
            self._predict_p = theano.function(inputs=self.x, outputs=self.softmax_p)
        return self._predict_p(x)

    def predict_p_and_v(self, x):
        if not hasattr(self, '_predict_p_and_v'):
            self._predict_p_and_v = theano.function(inputs=self.x, outputs=[self.softmax_p, self.logits_v])
        global mutex
        mutex.acquire()
        p, v = self._predict_p_and_v(x)
        mutex.release()
        return p, v

    def train(self, x, y_r, a):
        if not hasattr(self, '_train'):
            self._compute = theano.function(inputs=[self.x, self.y_r, self.action_index],
                                            outputs=self.cost_all)
            self._train = theano.function(updater=self.opt)
        global mutex
        mutex.acquire()
        loss =  self._compute(x, y_r, a)
        mutex.release()
        self._train()
        return loss

    def save(self, episode):
        filename = 'checkpoints/%s_%08d' % (self.model_name, episode)
        ws.Snapshot(self.network_params, filename)

    def load(self):
        filename = 'checkpoints/%s_%08d.bin' % (self.model_name, Config.LOAD_EPISODE)
        ws.Restore(filename)
        return Config.LOAD_EPISODE