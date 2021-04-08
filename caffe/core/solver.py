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
"""Optimization solvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import time

import google.protobuf.text_format

from dragon.core.autograph import function_impl
from dragon.core.training.adam import Adam
from dragon.core.training.rmsprop import RMSprop
from dragon.core.training.sgd import SGD
from dragon.core.training.sgd import Nesterov
from dragon.core.util import logging
from dragon.vm.caffe.core.net import Net
from dragon.vm.caffe.core.proto import caffe_pb2


class Solver(object):
    """Base solver class to optimize parameters."""

    def __init__(self, solver_file, is_root=True):
        """Create a ``Solver``.

        Parameters
        ----------
        solver_file : str
            The path of text proto file to load solver.
        is_root : bool, optional, default=True
            ``True`` to indicate a root solver.

        """
        self._is_root = is_root
        self._proto = caffe_pb2.SolverParameter()
        with open(solver_file, 'r') as f:
            google.protobuf.text_format.Parse(f.read(), self._proto)
        if self._proto.iter_size > 1:
            raise NotImplementedError('Gradient accumulation is not supported.')
        self._optimizer_args = {
            'scale': 1. / self._proto.iter_size,
            'clip_norm': float(self._proto.clip_gradients),
            'weight_decay': float(self._proto.weight_decay)
            if str(self._proto.regularization_type) == 'L2' else 0,
        }
        self._optimizer = None
        self._net = None
        self._test_nets = []
        self._iter = 0
        self._current_step = 0
        self._init_train_net()
        self._init_test_nets()

    @property
    def base_lr(self):
        """Return or Set the current learning rate.

        Returns
        -------
        float
            The current learning rate.

        """
        return self._optimizer.lr

    @base_lr.setter
    def base_lr(self, value):
        """Set the current learning rate.

        Parameters
        ----------
        value : float
            The value of learning rate to set.

        """
        self._optimizer.lr = value

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
            % (self._proto.snapshot_prefix, self._iter))

    def step(self, num_iterations=1):
        """Step the train net.

        Parameters
        ----------
        num_iterations : int, optional, default=1
            The number of iterations to step.

        """
        end_iter = self.iter + num_iterations
        tic = time.time()
        while self.iter < end_iter:
            # Test if necessary.
            if (self._proto.test_interval > 0 and
                    self.iter % self._proto.test_interval == 0):
                if (self.iter == 0 and self._proto.test_initialization) or self.iter != 0:
                    for test_idx in range(len(self._test_nets)):
                        self.test(test_idx)
            # Forward and backward pass.
            self._net.forward()
            # Display iteration info.
            if self._is_root and self._proto.display:
                if self.iter % self._proto.display == 0:
                    logging.info('Iteration %d, lr = %f, time = %.2fs'
                                 % (self.iter, self.base_lr, time.time() - tic))
                    tic = time.time()
                    for i, key in enumerate(self.net.outputs):
                        value = self.net.blobs[key].data.numpy().mean()
                        logging.info(' ' * 4 + 'Train net output #{}({}): {}'
                                     .format(i, key, value))
            # Apply Update.
            self._get_learning_rate()
            self._apply_update()
            self.iter = self.iter + 1
            # Snapshot if necessary.
            if self._proto.snapshot:
                if self.iter % self._proto.snapshot == 0:
                    self.snapshot()

    def test(self, test_idx):
        """Test the specific net.

        Parameters
        ----------
        test_idx : int
            The index of test net.

        """
        net = self._test_nets[test_idx]
        net.copy_from(self._net.to_proto())
        test_iter = self._proto.test_iter[test_idx]
        test_scores = collections.defaultdict(float)
        for iter in range(test_iter):
            net.forward()
            for key in net.outputs:
                test_scores[key] += float(net.blobs[key].data.numpy().mean())
        logging.info('Iteration {}, Test net #{}'.format(self.iter, test_idx))
        for i, (key, score) in enumerate(test_scores.items()):
            logging.info(' ' * 4 + 'Test net output #%d(%s): %.4f'
                         % (i, key, score / test_iter))

    def _init_train_net(self):
        """Initialize the train net."""
        if self._proto.HasField('net'):
            self._net = Net(self._proto.net, 'TRAIN')
        if self._proto.HasField('train_net'):
            if self._net is not None:
                raise RuntimeError('Specify either <net> or <train_net>.')
            self._net = Net(self._proto.train_net, 'TRAIN')

    def _init_test_nets(self):
        """Initialize the test nets."""
        if not self._is_root:
            # Only the root solver can do testing.
            return
        num_test_net = len(self._proto.test_iter)
        if num_test_net > 0:
            if self._proto.test_interval <= 0:
                raise RuntimeError('Test interval is invalid.')
        if len(self._proto.test_net) > 0:
            for test_net in self._proto.test_net:
                self._test_nets.append(Net(test_net, 'TEST'))
            num_test_net -= len(self._proto.test_net)
        if num_test_net > 0:
            self._test_nets.append(Net(self._proto.net, 'TEST'))

    def _get_learning_rate(self):
        """Get the learning rate based on preset policy."""
        policy = self._proto.lr_policy
        if policy == "step":
            new_step = int(self.iter / self._proto.stepsize)
            if self._current_step != new_step:
                new_lr = self._proto.base_lr * pow(self._proto.gamma, new_step)
                self._current_step = new_step
                self.base_lr = new_lr
        elif policy == 'multistep':
            if self._current_step < len(self._proto.stepvalue) \
                    and self.iter >= self._proto.stepvalue[self._current_step]:
                self._current_step = self._current_step + 1
                logging.info(
                    'MultiStep Status: Iteration {}, step = {}'
                    .format(self.iter, self._current_step))
                new_lr = (self._proto.base_lr *
                          pow(self._proto.gamma, self._current_step))
                self.base_lr = new_lr
        elif policy == 'multifixed':
            stage_lrs = self._proto.stage_lr
            stage_iters = self._proto.stage_iter
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
            power = self._proto.power
            gamma = self._proto.gamma
            self.base_lr = (self._proto.base_lr *
                            pow(1. + gamma * self.iter, -power))
        elif policy == 'poly':
            power = self._proto.power
            max_iter = self._proto.max_iter
            self.base_lr = (self._proto.base_lr *
                            pow(1. - float(self.iter) / max_iter, power))

    @function_impl.function
    def _apply_update(self):
        """Apply the weights update."""
        grads_and_vars = [(blob.diff, blob.data)
                          for blob in self._net._learnable_blobs]
        return self._optimizer.apply_gradients(grads_and_vars)


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
    }
    ```

    """

    def __init__(self, solver_file, is_root=True):
        """Create a ``AdamSolver``.

        Parameters
        ----------
        solver_file : str
            The path of solver file.
        is_root : bool, optional, default=True
            ``True`` to indicate a root solver.

        """
        super(AdamSolver, self).__init__(solver_file, is_root)
        self._optimizer_args['lr'] = self._proto.base_lr
        self._optimizer_args['beta1'] = self._proto.momentum
        self._optimizer_args['beta2'] = self._proto.momentum2
        self._optimizer_args['eps'] = self._proto.delta
        self._optimizer = Adam(**self._optimizer_args)


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
            ``True`` to indicate a root solver.

        """
        super(NesterovSolver, self).__init__(solver_file, is_root)
        self._optimizer_args['lr'] = self._proto.base_lr
        self._optimizer_args['momentum'] = self._proto.momentum
        self._optimizer = Nesterov(**self._optimizer_args)


class RMSPropSolver(Solver):
    r"""The RMSProp solver.
    `[Hinton et.al, 2013] <http://www.cs.utoronto.ca/~bonner/courses/2016s/csc321/lectures/lec6.pdf>`_.

    Examples:

    ```python
    solver {
      base_lr=0.01,
      rms_decay=0.99,
      delta=1e-8,
    }
    ```

    """

    def __init__(self, solver_file, is_root=True):
        """Create a ``RMSPropSolver``.

        Parameters
        ----------
        solver_file : str
            The path of solver file.
        is_root : bool, optional, default=True
            ``True`` to indicate a root solver.

        """
        super(RMSPropSolver, self).__init__(solver_file, is_root)
        self._optimizer_args['lr'] = self._proto.base_lr
        self._optimizer_args['decay'] = self._proto.rms_decay
        self._optimizer_args['eps'] = self._proto.delta
        self._optimizer = RMSprop(**self._optimizer_args)


class SGDSolver(Solver):
    r"""The Momentum-SGD solver.
    `[Polyak, 1964] <https://doi.org/10.1016/0041-5553(64)90137-5>`_.

    Examples:

    ```python
    solver {
      base_lr=0.01,
      momentum=0.9,
    }
    ```

    """

    def __init__(self, solver_file, is_root=True):
        """Create a ``SGDSolver``.

        Parameters
        ----------
        solver_file : str
            The path of solver file.
        is_root : bool, optional, default=True
            ``True`` to indicate a root solver.

        """
        super(SGDSolver, self).__init__(solver_file, is_root)
        self._optimizer_args['lr'] = self._proto.base_lr
        self._optimizer_args['momentum'] = self._proto.momentum
        self._optimizer = SGD(**self._optimizer_args)
