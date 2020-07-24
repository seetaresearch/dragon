# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""The solver to update parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from google.protobuf import text_format

from dragon.core.autograph import def_function
from dragon.core.training.adam import Adam
from dragon.core.training.rmsprop import RMSprop
from dragon.core.training.sgd import SGD
from dragon.core.training.sgd import Nesterov
from dragon.core.util import logging
from dragon.vm.caffe.core.net import Net
from dragon.vm.caffe.core.proto import caffe_pb2


class Solver(object):
    """The abstraction ``caffe.Solver``."""

    def __init__(self, solver_file, is_root=True):
        """Create a ``Solver``.

        Parameters
        ----------
        solver_file : str
            The path of solver file.
        is_root : bool, optional, default=True
            **True** to indicate a root solver.

        """
        self._is_root = is_root
        self._param = caffe_pb2.SolverParameter()
        with open(solver_file, 'r') as f:
            text_format.Parse(f.read(), self._param)
        if self._param.iter_size > 1:
            raise NotImplementedError('GradientAccum is deprecated.')
        self._arguments = {
            'scale': 1. / self._param.iter_size,
            'clip_norm': float(self._param.clip_gradients),
            'weight_decay': float(self._param.weight_decay)
            if str(self._param.regularization_type) == 'L2' else 0,
        }
        self._optimizer = None
        self._net, self._test_nets = None, []
        self._iter = self._current_step = 0
        self._init_train_net()
        self._init_test_nets()

    def _init_train_net(self):
        """Initialize the train net."""
        if self._param.HasField('net'):
            self._net = Net(self._param.net, 'TRAIN')
        if self._param.HasField('train_net'):
            if self._net is not None:
                raise RuntimeError('Specify either <net> or <train_net>.')
            self._net = Net(self._param.train_net, 'TRAIN')

    def _init_test_nets(self):
        """Initialize the test nets."""
        if not self._is_root:
            # Only the root solver can do testing.
            return
        num_test_net = len(self._param.test_iter)
        if num_test_net > 0:
            if self._param.test_interval <= 0:
                raise RuntimeError('Test interval is invalid.')
        if len(self._param.test_net) > 0:
            for test_net in self._param.test_net:
                self._test_nets.append(Net(test_net, 'TEST'))
            num_test_net -= len(self._param.test_net)
        if num_test_net > 0:
            self._test_nets.append(Net(self._param.net, 'TEST'))

    def _get_learning_rate(self):
        """Get the learning rate based on preset policy."""
        policy = self._param.lr_policy
        if policy == "step":
            new_step = int(self.iter / self._param.stepsize)
            if self._current_step != new_step:
                new_lr = self._param.base_lr * pow(self._param.gamma, new_step)
                self._current_step = new_step
                self.base_lr = new_lr
        elif policy == 'multistep':
            if self._current_step < len(self._param.stepvalue) \
                    and self.iter >= self._param.stepvalue[self._current_step]:
                self._current_step = self._current_step + 1
                logging.info(
                    'MultiStep Status: Iteration {}, step = {}'
                    .format(self.iter, self._current_step))
                new_lr = self._param.base_lr * \
                    pow(self._param.gamma, self._current_step)
                self.base_lr = new_lr
        elif policy == 'multifixed':
            stage_lrs = self._param.stage_lr
            stage_iters = self._param.stage_iter
            if self.iter < stage_iters[self._current_step]:
                self.base_lr = stage_lrs[self._current_step]
            else:
                if self._current_step + 1 < len(stage_iters):
                    self._current_step = self._current_step + 1
                    logging.info(
                        'MultiFixed Status: Iteration {}, stage = {}'
                        .format(self.iter, self._current_step))
                    self.base_lr = stage_lrs[self._current_step]
        elif policy == 'inv':
            power = self._param.power
            gamma = self._param.gamma
            self.base_lr = self._param.base_lr * \
                pow(1. + gamma * self.iter, -power)
        elif policy == 'poly':
            power = self._param.power
            max_iter = self._param.max_iter
            self.base_lr = self._param.base_lr * \
                pow(1. - float(self.iter) / max_iter, power)

    @def_function.function
    def _apply_update(self):
        """Apply the weights update."""
        for blob in self.net._layer_blobs:
            if blob.lr_multiplier > 0 and blob.diff is not None:
                self._optimizer.apply_gradients(
                    values_and_grads=[(blob.data, blob.diff)],
                    lr_mult=blob.lr_multiplier,
                    decay_mult=blob.decay_multiplier,
                )
        return self._optimizer

    @property
    def base_lr(self):
        """Return or Set the current learning rate.

        Returns
        -------
        float
            The current learning rate.

        """
        return self._optimizer.base_lr

    @base_lr.setter
    def base_lr(self, value):
        """Set the current learning rate.

        Parameters
        ----------
        value : float
            The value of learning rate to set.

        """
        self._optimizer.base_lr = value

    @property
    def iter(self):
        """Return or Set the current iteration.

        Returns
        -------
        int
            The current iteration.

        """
        return self._iter

    @iter.setter
    def iter(self, value):
        """Set the current iteration.

        Parameters
        ----------
        value : int
            The value of iteration to set.

        """
        self._iter = value

    @property
    def net(self):
        """Return the train net.

        Returns
        -------
        dragon.vm.caffe.Net
            The train net.

        """
        return self._net

    @property
    def test_nets(self):
        """Return the test nets.

        Returns
        -------
        Sequence[dragon.vm.caffe.Net]
            The test nets.

        """
        return self._test_nets

    def snapshot(self):
        """Snapshot the parameters of train net."""
        self._net.save(
            '%s_iter_%d.caffemodel'
            % (self._param.snapshot_prefix, self._iter))

    def step(self, num_iterations=1):
        """Step the train net.

        Parameters
        ----------
        num_iterations : int, optional, default=1
            The number of iterations to step.

        """
        start_step = self.iter
        stop_step = start_step + num_iterations
        loss_vec, smoothed_loss = [], 0.

        tic = time.time()
        while self.iter < stop_step:
            # Test if necessary.
            if self._is_root and self._param.test_interval > 0 and \
                    self.iter % self._param.test_interval == 0:
                if (self.iter == 0 and self._param.test_initialization) or \
                        self.iter != 0:
                    for test_idx in range(len(self._test_nets)):
                        self.test(test_idx)
            # Forward, backward and compute loss.
            loss = 0.
            for i in range(self._param.iter_size):
                self._net.forward_backward()
                if self._is_root:
                    for e in self.net.losses:
                        values = e.get_value().flatten()
                        for v in values:
                            loss += v
            if self._is_root:
                loss /= self._param.iter_size
                if len(loss_vec) < self._param.average_loss:
                    loss_vec.append(loss)
                    smoothed_loss *= (len(loss_vec) - 1)
                    smoothed_loss += loss
                    smoothed_loss /= len(loss_vec)
                else:
                    idx = (self.iter - start_step) % self._param.average_loss
                    smoothed_loss += ((loss - loss_vec[idx]) / self._param.average_loss)
                    loss_vec[idx] = loss
            # Apply Update.
            self._get_learning_rate()
            self._apply_update()
            # Display iteration info.
            if self._is_root and self._param.display:
                if self.iter % self._param.display == 0:
                    logging.info(
                        'Iteration %d, lr = %s, loss = %f, time = %.2fs'
                        % (self.iter, str(self.base_lr), smoothed_loss, time.time() - tic))
                    tic = time.time()
                    for idx, net_output in enumerate(self.net.outputs):
                        values = self.net.blobs[net_output].data.get_value().flatten()
                        for v in values:
                            logging.info(
                                ' ' * 10 + 'Train net output #{}({}): {}'
                                .format(idx, net_output, v))
            self.iter = self.iter + 1
            # Snapshot if necessary.
            if self._param.snapshot:
                if self.iter % self._param.snapshot == 0:
                    self.snapshot()

    def test(self, test_idx):
        """Test the specific net.

        Parameters
        ----------
        test_idx : int
            The idx of test net.

        """
        test_score, output_id = [], []
        net = self._test_nets[test_idx]
        test_iter = self._param.test_iter[test_idx]

        for iter in range(test_iter):
            net.forward()
            if not self._is_root:
                continue
            if iter == 0:
                for key in net.outputs:
                    values = net.blobs[key].data.get_value().flatten()
                    for idx, value in enumerate(values):
                        test_score.append(value)
                        output_id.append(key)
            else:
                i = 0
                for key in net.outputs:
                    values = net.blobs[key].data.get_value().flatten()
                    for idx, value in enumerate(values):
                        test_score[i] += value
                        i += 1

        logging.info('Iteration {}, Test net #{}'.format(self.iter, test_idx))
        for i, score in enumerate(test_score):
            logging.info(
                ' ' * 10 + 'Test net output #%d(%s): %.4f'
                % (i, output_id[i], score / test_iter))


class AdamSolver(Solver):
    r"""The Adam solver.
    `[Kingma & Ba, 2014] <https://arxiv.org/abs/1412.6980>`_.

    Examples:

    ```python
    solver {
      base_lr=0.001,
      momentum=0.9,
      momentum2=0.999,
      delta=1e-8,
    )
    ```

    """

    def __init__(self, solver_file, is_root=True):
        """Create a ``AdamSolver``.

        Parameters
        ----------
        solver_file : str
            The path of solver file.
        is_root : bool, optional, default=True
            **True** to indicate a root solver.

        """
        super(AdamSolver, self).__init__(solver_file, is_root)
        self._arguments['base_lr'] = self._param.base_lr
        self._arguments['beta1'] = self._param.momentum
        self._arguments['beta2'] = self._param.momentum2
        self._arguments['eps'] = self._param.delta
        self._optimizer = Adam(**self._arguments)


class NesterovSolver(Solver):
    r"""The Nesterov-SGD solver.
    `[Sutskever et.al, 2013] <http://www.cs.toronto.edu/~hinton/absps/momentum.pdf>`_.

    Examples:

    ```python
    solver {
      base_lr: 0.01
      momentum: 0.9
    }
    ```

    """

    def __init__(self, solver_file, is_root=True):
        """Create a ``NesterovSolver``.

        Parameters
        ----------
        solver_file : str
            The path of solver file.
        is_root : bool, optional, default=True
            **True** to indicate a root solver.

        """
        super(NesterovSolver, self).__init__(solver_file, is_root)
        self._arguments['base_lr'] = self._param.base_lr
        self._arguments['momentum'] = self._param.momentum
        self._optimizer = Nesterov(**self._arguments)


class RMSPropSolver(Solver):
    r"""The RMSProp solver.
    `[Hinton et.al, 2013] <http://www.cs.utoronto.ca/~bonner/courses/2016s/csc321/lectures/lec6.pdf>`_.

    Examples:

    ```python
    solver {
      base_lr=0.01,
      rms_decay=0.99,
      delta=1e-8,
    )
    ```

    """

    def __init__(self, solver_file, is_root=True):
        """Create a ``RMSPropSolver``.

        Parameters
        ----------
        solver_file : str
            The path of solver file.
        is_root : bool, optional, default=True
            **True** to indicate a root solver.

        """
        super(RMSPropSolver, self).__init__(solver_file, is_root)
        self._arguments['base_lr'] = self._param.base_lr
        self._arguments['decay'] = self._param.rms_decay
        self._arguments['eps'] = self._param.delta
        self._optimizer = RMSprop(**self._arguments)


class SGDSolver(Solver):
    r"""The Momentum-SGD solver.
    `[Polyak, 1964] <https://doi.org/10.1016/0041-5553(64)90137-5>`_.

    Examples:

    ```python
    solver {
      base_lr=0.01,
      momentum=0.9,
    )
    ```

    """

    def __init__(self, solver_file, is_root=True):
        """Create a `SGDSolver``.

        Parameters
        ----------
        solver_file : str
            The path of solver file.
        is_root : bool, optional, default=True
            **True** to indicate a root solver.

        """
        super(SGDSolver, self).__init__(solver_file, is_root)
        self._arguments['base_lr'] = self._param.base_lr
        self._arguments['momentum'] = self._param.momentum
        self._optimizer = SGD(**self._arguments)
