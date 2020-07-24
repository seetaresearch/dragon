# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Copyright (c) 2016-2018, The TensorLayer contributors.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

from dragon.core.framework import context
from dragon.core.framework import workspace
from dragon.core.util import logging
from dragon.vm.tensorlayer.core import initializers
from dragon.vm.tensorlayer.core import files


class Module(object):
    """The base class taking with neural-network features.

    ```python
    class MyModule(tl.Module):
        def __init__(name=None):
            super(MyModule, self).__init__(name)
    ```

    """

    def __init__(self, name=None):
        """Create a new ``Module``.

        Parameters
        ----------
        name : str, optional
            The optional module name.

        """
        self._name = name
        self._modules = []
        self._trainable_weights = []
        self._nontrainable_weights = []
        self._training = True

    @property
    def modules(self):
        """Return all the modules."""
        return self._filter_empty_containers(self._modules)

    @property
    def name(self):
        """Return the module name.

        Returns
        -------
        str
            The module name.

        """
        if self._name is None:
            self._set_name()
        return self._name

    @property
    def nontrainable_weights(self):
        """Return the non-trainable weights.

        Returns
        -------
        Sequence[dragon.Tensor]
            The weights sequence.

        """
        nested = self._gather_children_attribute('nontrainable_weights')
        return self._dedupe_weights(self._nontrainable_weights + nested)

    @property
    def training(self):
        """Return the training mode.

        Returns
        -------
        bool
            **True** for training otherwise evaluation.

        """
        return self._training

    @training.setter
    def training(self, mode):
        """Set the training mode."""
        self._training = mode
        for layer in self.modules:
            layer.training = mode

    @property
    def trainable_weights(self):
        """Return the trainable weights.

        Returns
        -------
        Sequence[dragon.Tensor]
            The weights sequence.

        """
        nested = self._gather_children_attribute('trainable_weights')
        return self._dedupe_weights(self._trainable_weights + nested)

    @property
    def weights(self):
        """Return all the weights, both trainable and non-trainable.

        Returns
        -------
        Sequence[dragon.Tensor]
            The weights sequence.

        """
        return self.trainable_weights + self.nontrainable_weights

    def add_weight(
        self,
        name=None,
        shape=None,
        init='glorot_uniform',
        trainable=True,
    ):
        """Add a new weight.

        Parameters
        ----------
        name : str, optional
            The weight name.
        shape : Sequence[int], optional
            The weight shape.
        init : Union[callable, str], optional
            The initializer for weight.
        trainable : bool, optional, default=True
            **True** to compute the gradients if necessary.

        Returns
        -------
        dragon.Tensor
            The weight tensor.

        """
        name = name if name else 'weights'
        shape = shape if shape is not None else []
        weight = initializers.get(init)(
            shape=shape,
            trainable=trainable,
            name=context.get_name_scope() + name,
        )
        if trainable is True:
            self._trainable_weights.append(weight)
        else:
            self._nontrainable_weights.append(weight)
        return weight

    def forward(self, *inputs, **kwargs):
        """Method to define the forward operations."""
        pass

    def load_weights(self, filepath, format=None, skip=False, verbose=False):
        """Load model weights from a binary file.

        Parameters
        ----------
        filepath : str
            The path of weights file.
        format : {'hdf5', 'npz', 'pkl', 'npz_dict'}, optional
            The optional saving format.
        skip : bool, optional, default=False
            **True** to skip the modules which is not found.
        verbose: bool, optional, default=False
            **True** to print the matched weights.

        """
        if not os.path.exists(filepath):
            raise FileNotFoundError("File {} doesn't exist.".format(filepath))

        if format is None:
            format = filepath.split('.')[-1]

        if format == 'hdf5' or format == 'h5':
            matched_info = files.load_hdf5_to_weights(filepath, self, skip)
        elif format == 'npz':
            matched_info = files.load_and_assign_npz(filepath, self)
        elif format == 'pkl':
            matched_info = files.load_and_assign_pkl_dict(filepath, self, skip)
        elif format == 'npz_dict':
            matched_info = files.load_and_assign_npz_dict(filepath, self, skip)
        else:
            raise ValueError(
                "Saving format should be in ('hdf5', 'npz', 'pkl', 'npz_dict').\n"
                "Format <%s> is not supported." % format
            )

        if verbose:
            for info in matched_info:
                logging.info(
                    'Weight({}) loaded, Size: ({})'
                    .format(info[0], ', '.join([str(d) for d in info[1]]))
                )

    def save_weights(self, filepath, format=None):
        """Save weights into a binary file.

        Parameters
        ----------
        filepath : str
            The path of weights file.
        format : {'hdf5', 'npz', 'pkl', 'npz_dict'}, optional
            The optional saving format.

        """
        if format is None:
            postfix = filepath.split('.')[-1]
            if postfix in ['h5', 'hdf5', 'npz']:
                format = postfix
            elif postfix in ['pkl', 'pkl_dict']:
                format = 'pkl'
            else:
                format = 'hdf5'
        if format == 'hdf5' or format == 'h5':
            files.save_weights_to_hdf5(filepath, self)
        elif format == 'npz':
            files.save_npz(self.weights, filepath)
        elif format == 'pkl':
            files.save_pkl_dict(self.weights, filepath)
        elif format == 'npz_dict':
            files.save_npz_dict(self.weights, filepath)
        else:
            raise ValueError(
                "Saving format should be in ('hdf5', 'npz', 'pkl', 'npz_dict').\n"
                "Format <%s> is not supported." % format
            )

    @staticmethod
    def _dedupe_weights(weights):
        """Dedupe weights according to identity."""
        output, seen_weights = [], set()
        for w in weights:
            wid = id(w)
            if wid not in seen_weights:
                output.append(w)
                seen_weights.add(wid)
        return output

    @staticmethod
    def _filter_empty_containers(containers):
        existing = set()
        to_visit = containers[::-1]
        filtered = []
        while to_visit:
            obj = to_visit.pop()
            obj_id = id(obj)
            if obj_id in existing:
                continue
            existing.add(obj_id)
            if hasattr(obj, '_modules'):
                filtered.append(obj)
                to_visit.extend(obj._modules[::-1])
        return filtered

    def _gather_children_attribute(self, attribute):
        assert attribute in {
            'all_weights',
            'trainable_weights',
            'nontrainable_weights',
        }
        return list(
            itertools.chain.from_iterable(
                getattr(m, attribute) for m in self.modules
            )
        )

    def _set_name(self, name=None, zero_based=True):
        """Set the module name."""
        if name is None:
            self._name = workspace.get_workspace().unique_name(
                name=self.__class__.__name__.lower(),
                namespace='Object',
                zero_based=zero_based,
            )
        else:
            self._name = name

    def __call__(self, *args, **kwargs):
        """The preprocessor for ``self.forward(...)``."""
        with context.name_scope(self.name):
            return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules.append(value)
            if value._name is None:
                # Use the attribute name instead of a dummy one.
                value._name = name
        # Add the attribute.
        object.__setattr__(self, name, value)
