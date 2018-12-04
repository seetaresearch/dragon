# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

import time

import dragon.core.workspace as ws
import dragon.core.mpi as mpi
import dragon.updaters as updaters
import dragon.tools.summary_writer as sw
import dragon.vm.theano as theano
from dragon.vm.caffe.proto import caffe_pb2 as pb

from dragon.vm.caffe.misc import root_solver
from dragon.vm.caffe.net import Net
from google.protobuf.text_format import Parse


class Solver(object):
    """
    Solver merges updates to optimize the ``Net``.

    Inspired by `SolverWrapper`_, we simplified it from the original C++ implementation.
    """
    def __init__(self, prototxt):
        """Construct a Solver.

        Parameters
        ----------
        prototxt : str
            The path of ``.prototxt`` file.

        Returns
        -------
        Solver
            The solver.

        Examples
        --------
        >>> solver = Solver('solver.prototxt')

        """
        self._param = pb.SolverParameter()
        Parse(open(prototxt, 'r').read(), self._param)
        self.ParseUpdateParam()
        self._net = None
        self._test_nets = []
        self._layer_blobs = []
        self._iter = self._current_step = 0
        self.optimizer = None
        self.scalar_writer = sw.ScalarSummary() if root_solver() else None

        self.InitTrainNet()
        self.InitTestNets()
        self.BuildNets()

    def InitTrainNet(self):
        """Initialize the train net.

        Returns
        -------
        None

        References
        ----------
        The implementation of `InitTrainNet(solver.cpp, L63)`_.

        """
        if self._param.HasField('net'):
            self._net = Net(self._param.net, "TRAIN")

        if self._param.HasField('train_net'):
            if self._net is not None:
                raise RuntimeError('net or train_net can not be specfied both.')
            self._net = Net(self._param.train_net, "TRAIN")

    def InitTestNets(self):
        """Initialize the test nets.

        Returns
        -------
        None

        References
        ----------
        The implementation of `InitTestNets(solver.cpp, L104)`_.

        """
        if mpi.Is_Init():
            idx, group = mpi.AllowParallel()
            # Only the root in a parallel group can test
            if idx != -1 and mpi.Rank() != group[0]: return

        num_test_net = len(self._param.test_iter)
        if num_test_net > 0:
            if self._param.test_interval <= 0:
                raise RuntimeError('the val of test interval: {} is invaild.')

        if len(self._param.test_net) > 0:
            for test_net in self._param.test_net:
                 self._test_nets.append(Net(test_net, "TEST"))
            num_test_net -= len(self._param.test_net)

        # Consider generic_net
        if num_test_net > 0:
            self._test_nets.append(Net(self._param.net, "TEST"))

        # Share with training net
        for test_net in self._test_nets: test_net.share_with(self._net)

    def BuildNets(self):
        """Build the nets.

        Returns
        -------
        None

        See Also
        --------
        `Net.function(*args, **kwargs)`_ - How to transform ``Net`` into ``Graph``.

        """
        self.train = self._net.function()
        self.tests = [test_net.function() for test_net in self._test_nets]

    def ParseUpdateParam(self):
        """Parse the parameters for optimizer.

        Returns
        ------=
        None

        """
        self._update_param = {
            'scale_gradient': float(1.0 / self._param.iter_size),
            'clip_gradient': float(self._param.clip_gradients),
            'l2_decay': float(self._param.weight_decay) \
                if str(self._param.regularization_type) == 'L2' else -1.0,
        }

    def BuildOptimizer(self):
        """Build the optimizer.

        Returns
        -------
        None

        """
        # Collect
        for layer, blobs in self._net.params.items():
            self._layer_blobs.extend(blobs)

        # Push
        for idx, blob in enumerate(self._layer_blobs):
            if self._net._lr_mults[idx] > 0:
                if blob.diff is None: continue
                self.optimizer.append(
                    (blob.data, blob.diff),
                        self._net._lr_mults[idx],
                            self._net._decay_mults[idx])

        # Compile
        self.update = theano.function(updater=self.optimizer)

    def GetLearningRate(self):
        """Get learning rate based on the preset policy.

        Returns
        -------
        None

        References
        ----------
        The implementation of `GetLearningRate(solver.cpp, L27)`_.

        """
        from dragon.config import logger
        policy = self._param.lr_policy

        if policy == "step":
            new_step = int(self._iter / self._param.stepsize)
            if self._current_step != new_step:
                new_lr = self._param.base_lr * pow(self._param.gamma, new_step)
                self._current_step = new_step
                self.optimizer.base_lr = new_lr

        if policy == 'multistep':
            if self._current_step < len(self._param.stepvalue) \
                    and self._iter >= self._param.stepvalue[self._current_step]:
                self._current_step = self._current_step + 1
                logger.info('MultiStep Status: Iteration {},  step = {}' \
                    .format(self._iter, self._current_step))
                new_lr = self._param.base_lr * \
                         pow(self._param.gamma, self._current_step)
                self.optimizer.base_lr = new_lr

        if policy == 'multifixed':
            stage_lrs = self._param.stage_lr
            stage_iters = self._param.stage_iter
            if self._iter < stage_iters[self._current_step]:
                self.optimizer.base_lr = stage_lrs[self._current_step]
            else:
                if self._current_step + 1 < len(stage_iters):
                    self._current_step = self._current_step + 1
                    logger.info('MultiFixed Status: Iteration {},  stage = {}' \
                        .format(self._iter, self._current_step))
                    self.optimizer.base_lr = stage_lrs[self._current_step]

        if policy == 'inv':
            power = self._param.power
            gamma = self._param.gamma
            self.optimizer.base_lr = self._param.base_lr * \
                pow(1.0 + gamma * self._iter, -power)

        if policy == 'poly':
            power = self._param.power
            max_iter = self._param.max_iter
            self.optimizer.base_lr = self._param.base_lr * \
                pow(1.0 - float(self.iter) / max_iter, power)

    def Test(self, test_idx):
        """Test the specific net.

        Parameters
        ----------
        test_idx : int
            The idx of test net.

        Returns
        -------
        None

        References
        ----------
        The implementation of `Test(solver.cpp, L328)`_.

        """
        from dragon.config import logger
        test_score = []
        output_id = []
        test_iter = self._param.test_iter[test_idx]
        net = self._test_nets[test_idx]

        for iter in range(test_iter):
            self.tests[test_idx](return_outputs=False)
            if not root_solver(): continue
            if iter == 0:
                for net_output in net._net_outputs:
                    values = ws.FetchTensor(net.blobs[net_output].data)
                    for idx, value in enumerate(values):
                        test_score.append(value)
                        output_id.append(net_output)
            else:
                i = 0
                for net_output in net._net_outputs:
                    values = ws.FetchTensor(net.blobs[net_output].data)
                    for idx, value in enumerate(values):
                        test_score[i] += value
                        i += 1

        if not root_solver(): return

        logger.info('Iteration {}, Test net #{}'.format(self._iter, test_idx))
        for idx, score in enumerate(test_score):
            logger.info('		 Test net output #%d(%s): %.4f' % (idx, output_id[idx], score / test_iter))
            self.scalar_writer.add_summary((output_id[idx], score / test_iter), self._iter)

    def step(self, iters):
        """Step the train net. [**PyCaffe Style**]

        Parameters
        ----------
        iters : int
            The number of iterations to step.

        Returns
        -------
        None

        References
        ----------
        The implementation of `Step(solver.cpp, L180)`_.

        """
        from dragon.config import logger
        start_iter, stop_iter = self._iter, self._iter + iters
        loss_vec, smoothed_loss = [], 0.

        tic = time.time()

        while self._iter < stop_iter:
            # Test if necessary
            if self._param.test_interval and \
                 self._iter % self._param.test_interval == 0:
                if (self._iter == 0 and
                        self._param.test_initialization) or self._iter != 0:
                    for test_id in range(len(self.tests)): self.Test(test_id)

            # Forward && Backward && Compute Loss
            loss = 0.0
            for i in range(self._param.iter_size):
                self.train(return_outputs=False)
                if root_solver():
                    for cost in self._net._costs:
                        cost_value = ws.FetchTensor(cost)
                        if cost_value.size == 1:
                            loss += cost_value[0]

            if root_solver():
                loss /= self._param.iter_size
                if len(loss_vec) < self._param.average_loss:
                    loss_vec.append(loss)
                    smoothed_loss = (smoothed_loss * (len(loss_vec) - 1) + loss) / len(loss_vec)
                else:
                    idx = (self._iter - start_iter) % self._param.average_loss
                    smoothed_loss += ((loss - loss_vec[idx]) / self._param.average_loss)
                    loss_vec[idx] = loss

            # Apply Update
            self.GetLearningRate()
            self.update()

            # Display
            if root_solver() and self._param.display:
                if self._iter % self._param.display == 0:
                    base_lr = self.optimizer.base_lr
                    logger.info('Iteration %d, lr = %s, loss = %f, time = %.2fs' % \
                          (self._iter, str(base_lr), smoothed_loss, time.time() - tic))
                    tic = time.time()
                    for idx, net_output in enumerate(self._net.outputs):
                        values = ws.FetchTensor(self._net.blobs[net_output].data)
                        for v in values:
                            logger.info('		Train net output #{}({}): {}'.format(idx, net_output, v))
                            self.scalar_writer.add_summary((net_output, v), self._iter)

            # Inc Iterations
            self._iter = self._iter + 1

            # Snapshot
            if self._param.snapshot:
                if self._iter % self._param.snapshot == 0: self.snapshot()

    def one_step(self):
        """One step run the train net.

        Returns
        -------
        dict
            The stats.

        """
        if self._param.test_interval and \
                self._iter % self._param.test_interval == 0:
            if (self._iter == 0 and
                    self._param.test_initialization) or self._iter != 0:
                for test_id in range(len(self.tests)): self.Test(test_id)

        # Forward && Backward && Compute_loss
        run_time, stats = 0., {'loss': {'total': 0.}, 'iter': self.iter}
        for i in range(self._param.iter_size):
            tic = time.time()
            self.train(return_outputs=False)
            run_time += (time.time() - tic)

            # Total loss
            for cost in self._net._costs:
                cost_value = ws.FetchTensor(cost)
                if cost_value.size == 1:
                    stats['loss']['total'] += cost_value[0]

            # Partial loss
            for idx, net_output in enumerate(self._net.outputs):
                values = ws.FetchTensor(self._net.blobs[net_output].data)
                if values.size != 1: continue
                if net_output not in stats['loss']: stats['loss'][net_output] = 0.
                stats['loss'][net_output] += values[0]

        # Apply Update
        self.GetLearningRate()
        tic = time.time()
        self.update()
        run_time += (time.time() - tic)
        self._iter = self._iter + 1

        # Snapshot
        if self._param.snapshot:
            if self._iter % self._param.snapshot == 0: self.snapshot()

        # Average loss by the iter size
        for k in stats['loss'].keys():
            stats['loss'][k] /= self._param.iter_size

        # Misc stats
        stats['lr'] = self.optimizer.base_lr
        stats['time'] = run_time
        return stats

    def snapshot(self):
        """Snapshot the parameters of train net. [**PyCaffe Style**]

        Returns
        -------
        None

        See Also
        --------
        `workspace.Snapshot(*args, **kwargs)`_ - How to snapshot tensors into a file.

        References
        ----------
        The implementation of `Snapshot(solver.cpp, L403)`_.

        """
        tensors = [blob.data for blob in self._layer_blobs]
        filename = "_iter_" + str(self._iter)
        ws.Snapshot(tensors, filename,
            prefix=self._param.snapshot_prefix,
                suffix='.caffemodel', format='caffe')

    @property
    def net(self):
        """Return the train net. [**PyCaffe Style**]

        Returns
        -------
        Net
            The train net.

        """
        return self._net

    @property
    def test_nets(self):
        """Return the test nets. [**PyCaffe Style**]

        Returns
        -------
        list of Net
            The test nets.

        """
        return self._test_nets

    @property
    def iter(self):
        """Return or Set the current iteration. [**PyCaffe Style**]

        Parameters
        ----------
        iter : int
            The value of iteration to set.

        Returns
        -------
            The current iteration.

        """
        return self._iter

    @iter.setter
    def iter(self, value):
        self._iter = value

    @property
    def base_lr(self):
        """Return or Set the current learning rate. [**Extended**]

        Parameters
        ----------
        iter : float
            The value of learning rate to set.

        Returns
        -------
        The current learning rate.

        """
        return self.optimizer.base_lr

    @base_lr.setter
    def base_lr(self, value):
        self.optimizer.base_lr = value


class SGDSolver(Solver):
    """The Momentum-SGD Solver, introduced by `[LeCun et.al, 1998]`_.

    Parameters
    ----------
    base_lr : float
        Refer `SolverParameter.base_lr`_.
    momentum : float
        Refer `SolverParameter.momentum`_.

    """
    def __init__(self, prototxt):
        super(SGDSolver, self).__init__(prototxt=prototxt)
        self.optimizer = updaters.SGDUpdater(**self._update_param)
        self.BuildOptimizer()

    def ParseUpdateParam(self):
        super(SGDSolver, self).ParseUpdateParam()
        self._update_param['base_lr'] = self._param.base_lr
        self._update_param['momentum'] = self._param.momentum


class NesterovSolver(Solver):
    """The Nesterov-SGD Solver, introduced by `[Sutskever et.al, 2012]`_.

    Parameters
    ----------
    base_lr : float
        Refer `SolverParameter.base_lr`_.
    momentum : float
        Refer `SolverParameter.momentum`_.

    """
    def __init__(self, prototxt):
        super(NesterovSolver, self).__init__(prototxt=prototxt)
        self.optimizer = updaters.NesterovUpdater(**self._update_param)
        self.BuildOptimizer()

    def ParseUpdateParam(self):
        super(NesterovSolver, self).ParseUpdateParam()
        self._update_param['base_lr'] = self._param.base_lr
        self._update_param['momentum'] = self._param.momentum


class RMSPropSolver(Solver):
    """The RMSProp Solver, introduced by `[Hinton et.al, 2013]`_.

    Parameters
    ----------
    base_lr : float
        Refer `SolverParameter.base_lr`_.
    rms_decay : float
        Refer `SolverParameter.rms_decay`_.
    delta : float
        Refer `SolverParameter.delta`_.

    """
    def __init__(self, prototxt):
        super(RMSPropSolver, self).__init__(prototxt=prototxt)
        self.optimizer = updaters.RMSPropUpdater(**self._update_param)
        self.BuildOptimizer()

    def ParseUpdateParam(self):
        super(RMSPropSolver, self).ParseUpdateParam()
        self._update_param['base_lr'] = self._param.base_lr
        self._update_param['decay'] = self._param.rms_decay
        self._update_param['eps'] = self._param.delta


class AdamSolver(Solver):
    """The Adam Solver, introduced by `[Kingma & Ba, 2014]`_.

    Parameters
    ----------
    base_lr : float
        Refer `SolverParameter.base_lr`_.
    momentum : float
        Refer `SolverParameter.momentum`_.
    momentum2 : float
        Refer `SolverParameter.momentum2`_.
    delta : float
        Refer `SolverParameter.delta`_.

    """
    def __init__(self, prototxt):
        super(AdamSolver, self).__init__(prototxt=prototxt)
        self.optimizer = updaters.AdamUpdater(**self._update_param)
        self.BuildOptimizer()

    def ParseUpdateParam(self):
        super(AdamSolver, self).ParseUpdateParam()
        self._update_param['base_lr'] = self._param.base_lr
        self._update_param['beta1'] = self._param.momentum
        self._update_param['beta2'] = self._param.momentum2
        self._update_param['eps'] = self._param.delta