# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import time

import dragon.core.mpi as mpi
import dragon.updaters as updaters
import dragon.tools.summary_writer as sw
import proto.caffe_pb2 as pb
from dragon.core.workspace import FetchTensor, Snapshot
from dragon.vm.caffe.common import root_solver
from dragon.vm.caffe.net import Net
from dragon.vm.theano import function
from google.protobuf.text_format import Parse

class Solver(object):
    def __init__(self, prototxt):
        self._param = pb.SolverParameter()
        Parse(open(prototxt, 'r').read(), self._param)
        self._net = None; self._test_nets = []
        self._iter = self._current_step = 0
        self.train = self.tests = self.update = self._updater = None
        self.scalar_writer = sw.ScalarSummary() if root_solver() else None
        self._lr_blobs = []
        self.InitTrainNet()
        self.InitTestNets()
        self.CheckUpdateParam()


    def InitTrainNet(self):
        if self._param.HasField('net'):
            self._net = Net(self._param.net, "TRAIN")

        if self._param.HasField('train_net'):
            if self._net is not None:
                raise RuntimeError('net or train_net can not be specfied both.')
            self._net = Net(self._param.train_net, "TRAIN")


    def InitTestNets(self):
        if mpi.is_init():
            idx, group = mpi.allow_parallel()
            # only the root in a parallel group can test
            if idx != -1 and mpi.rank() != group[0]: return

        num_test_net = len(self._param.test_iter)
        if num_test_net > 0:
            if self._param.test_interval <= 0:
                raise RuntimeError('the val of test interval: {} is invaild.')

        if len(self._param.test_net) > 0:
            for test_net in self._param.test_net:
                 self._test_nets.append(Net(test_net, "TEST"))
            num_test_net -= len(self._param.test_net)

        # consider generic_net
        if num_test_net > 0:
            self._test_nets.append(Net(self._param.net, "TEST"))

        # share with training net
        for test_net in self._test_nets: test_net.share_with(self._net)


    def step(self, iters):
        """ simply follow the pycaffe style """
        start_iter = self._iter; stop_iter = self._iter + iters
        loss_vec = []; smoothed_loss = 0
        tic = time.time()
        while self._iter < stop_iter:
            # test if necessary
            if self._param.test_interval and \
                 self._iter % self._param.test_interval == 0:
                if (self._iter == 0 and
                        self._param.test_initialization) or self._iter != 0:
                    for test_id in xrange(len(self.tests)): self.Test(test_id)

            # forward & backward & compute_loss
            loss = 0.0
            for i in xrange(self._param.iter_size):
                self.train(return_outputs=False)
                if root_solver():
                    for cost in self._net._costs: loss += FetchTensor(cost)[0]

            if root_solver():
                loss /= self._param.iter_size
                if len(loss_vec) < self._param.average_loss:
                    loss_vec.append(loss)
                    smoothed_loss = (smoothed_loss * (len(loss_vec) - 1) + loss) / len(loss_vec);
                else:
                    idx = (self._iter - start_iter) % self._param.average_loss
                    smoothed_loss += ((loss - loss_vec[idx]) / self._param.average_loss)
                    loss_vec[idx] = loss

            # apply update
            self.CheckLearningRate()
            self.update()

            # display
            if root_solver() and self._param.display:
                if self._iter % self._param.display == 0:
                    base_lr = self._updater.lr
                    print 'Iteration %d, lr = %s, loss = %f, time = %.2fs' % \
                          (self._iter, str(base_lr), smoothed_loss, time.time() - tic)
                    tic = time.time()
                    for idx, net_output in enumerate(self._net._net_outputs):
                        vals = FetchTensor(self._net.blobs[net_output].data)
                        for val in vals:
                            print '		Train net output #{}({}): {}'.format(idx, net_output, val)
                            self.scalar_writer.add_summary((net_output, val), self._iter)
            self._iter = self._iter + 1

            # snapshot
            if self._param.snapshot:
                if self._iter % self._param.snapshot == 0: self.snapshot()


    def Test(self, test_idx):
        test_score = []; output_id = []
        test_iter = self._param.test_iter[test_idx]
        net = self._test_nets[test_idx]
        for iter in xrange(test_iter):
            self.tests[test_idx](return_outputs=False)
            if not root_solver(): continue
            if iter == 0:
                for net_output in net._net_outputs:
                    vals = FetchTensor(net.blobs[net_output].data)
                    for idx, val in enumerate(vals):
                        test_score.append(val)
                        output_id.append(net_output)
            else:
                i = 0
                for net_output in net._net_outputs:
                    vals = FetchTensor(net.blobs[net_output].data)
                    for idx, val in enumerate(vals):
                        test_score[i] += val; i = i + 1
        if not root_solver(): return
        print 'Iteration {}, Test net #{}'.format(self._iter, test_idx)
        for idx, score in enumerate(test_score):
            print '		 Test net output #%d(%s): %.4f' % (idx, output_id[idx], score / test_iter)
            self.scalar_writer.add_summary((output_id[idx], score / test_iter), self._iter)


    def CheckUpdateParam(self):
        self._update_param = {'scale_gradient': float(1.0 / self._param.iter_size),
                              'clip_gradient': float(self._param.clip_gradients),
                              'l2_decay': float(self._param.weight_decay)
                              if str(self._param.regularization_type) == 'L2' else -1.0}


    def CheckLearningRate(self):
        policy = self._param.lr_policy

        if policy == "step":
            new_step = int(self._iter / self._param.stepsize)
            if self._current_step != new_step:
                new_lr = self._param.base_lr * pow(self._param.gamma, new_step)
                self._current_step = new_step
                self._updater.lr = new_lr

        if policy == 'multistep':
            if self._current_step < len(self._param.stepvalue) \
                    and self._iter >= self._param.stepvalue[self._current_step]:
                self._current_step = self._current_step + 1
                print 'MultiStep Status: Iteration {},  step = {}' \
                    .format(self._iter, self._current_step)
                new_lr = self._param.base_lr * \
                         pow(self._param.gamma, self._current_step)
                self._updater.lr = new_lr

        if policy == 'multifixed':
            stage_lrs = self._param.stage_lr
            stage_iters = self._param.stage_iter
            if self._iter < stage_iters[self._current_step]:
                self._updater.lr = stage_lrs[self._current_step]
            else:
                if self._current_step + 1 < len(stage_iters):
                    self._current_step = self._current_step + 1
                    print 'MultiFixed Status: Iteration {},  stage = {}' \
                        .format(self._iter, self._current_step)
                    self._updater.lr = stage_lrs[self._current_step]

        if policy == 'inv':
            power = self._param.power
            gamma = self._param.gamma
            self._updater.lr = self._param.base_lr * \
                               pow(1.0 + gamma * self._iter, -power)

        if policy == 'poly':
            power = self._param.power
            max_iter = self._param.max_iter
            self._updater.lr = self._param.base_lr * \
                        pow(1.0 - float(self.iter) / max_iter, power)


    def snapshot(self):
        """ simply follow the pycaffe style """
        tensors = [blob.data for blob in self._lr_blobs]
        filename = "_iter_" + str(self._iter)
        Snapshot(tensors, filename, prefix=self._param.snapshot_prefix,
                 suffix='.caffemodel', format=1)

    @property
    def net(self):
        return self._net


    @property
    def test_nets(self):
        return self._test_nets


    @property
    def iter(self):
        return self._iter


class SGDSolver(Solver):
    def __init__(self, prototxt):
        super(SGDSolver, self).__init__(prototxt=prototxt)
        self._updater = updaters.SGDUpdater(**self._update_param)

        # generates update targets
        for layer, blobs in self._net.params.iteritems():  self._lr_blobs.extend(blobs)
        for idx, blob in enumerate(self._lr_blobs):
            if self._net._lr_mults[idx] > 0:
                if blob.diff is None: continue
                self._updater.append((blob.data, blob.diff),
                                     self._net._lr_mults[idx], self._net._decay_mults[idx])
        self.train = self._net.function
        self.tests = [test_net.function for test_net in self._test_nets]
        self.update = function(updater=self._updater)


    def CheckUpdateParam(self):
        super(SGDSolver, self).CheckUpdateParam()
        params = ['base_lr', 'momentum']
        for param in params:
            if self._param.HasField(param):
                self._update_param[param] = getattr(self._param, param)

class RMSPropSolver(Solver):
    def __init__(self, prototxt):
        super(RMSPropSolver, self).__init__(prototxt=prototxt)
        self._updater = updaters.RMSPropUpdater(**self._update_param)

        # generates update targets
        for layer, blobs in self._net.params.iteritems():  self._lr_blobs.extend(blobs)
        for idx, blob in enumerate(self._lr_blobs):
            if self._net._lr_mults[idx] > 0:
                if blob.diff is None: continue
                self._updater.append((blob.data, blob.diff),
                                     self._net._lr_mults[idx], self._net._decay_mults[idx])
        self.train = self._net.function
        self.tests = [test_net.function for test_net in self._test_nets]
        self.update = function(updater=self._updater)


    def CheckUpdateParam(self):
        super(RMSPropSolver, self).CheckUpdateParam()
        self._update_param['base_lr'] = self._param.base_lr
        self._update_param['decay'] = self._param.rms_decay
        self._update_param['eps'] = self._param.delta

class AdamSolver(Solver):
    def __init__(self, prototxt):
        super(AdamSolver, self).__init__(prototxt=prototxt)
        self._updater = updaters.AdamUpdater(**self._update_param)

        # generates update targets
        for layer, blobs in self._net.params.iteritems():  self._lr_blobs.extend(blobs)
        for idx, blob in enumerate(self._lr_blobs):
            if self._net._lr_mults[idx] > 0:
                if blob.diff is None: continue
                self._updater.append((blob.data, blob.diff),
                                     self._net._lr_mults[idx], self._net._decay_mults[idx])
        self.train = self._net.function
        self.tests = [test_net.function for test_net in self._test_nets]
        self.update = function(updater=self._updater)


    def CheckUpdateParam(self):
        super(AdamSolver, self).CheckUpdateParam()
        self._update_param['base_lr'] = self._param.base_lr
        self._update_param['beta1'] = self._param.momentum
        self._update_param['beta2'] = self._param.momentum2
        self._update_param['eps'] = self._param.delta