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
"""Base layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from dragon.core.framework import context
from dragon.core.util import nest
from dragon.vm.keras.core import initializers
from dragon.vm.keras.core import regularizers
from dragon.vm.keras.core import saving
from dragon.vm.keras.core.engine import input_spec
from dragon.vm.tensorflow.core.framework import dtypes
from dragon.vm.tensorflow.core.module import module
from dragon.vm.tensorflow.core.ops.variables import Variable


class Layer(module.Module):
    """The base class of layers.

    Inherit this class to design a new layer:

    ```python
    class MyLayer(tf.keras.layers.Layer):
        def __init__():
            super(MyModule, self).__init__()
    ```

    """

    def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
        """Create a ``Layer``.

        Parameters
        ----------
        trainable : bool, optional, default=True
            The initial training flag.
        name : str, optional
            The layer name.
        dtype : dragon.vm.tensorflow.dtypes.DType
            The optional data type.

        """
        super(Layer, self).__init__(name)
        self._trainable = trainable
        self._dtype = dtype or dtypes.float32
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._input_spec = None
        self.built = False

    @property
    def dtype(self):
        """Return the data type.

        Returns
        -------
        str
            The data type.

        """
        return self._dtype

    @property
    def non_trainable_weights(self):
        """Return the non-trainable weights.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.Variable]
            The non-trainable weights.

        """
        if self.trainable:
            nested = self._gather_children_attribute('non_trainable_variables')
            weights = self._non_trainable_weights + nested
        else:
            weights = (self._trainable_weights +
                       self._non_trainable_weights +
                       self._gather_children_attribute('variables'))
        return self._dedupe_weights(weights)

    @property
    def input_spec(self):
        """Return the input spec.

        Returns
        -------
        Union[InputSpec, Sequence[InputSpec]]
            The input spec.

        """
        return self._input_spec

    @input_spec.setter
    def input_spec(self, value):
        """Return the input spec.

        Parameters
        ----------
        value : Union[InputSpec, Sequence[InputSpec]]
            The input spec to set.

        """
        for v in nest.flatten(value):
            if v is not None and not isinstance(v, input_spec.InputSpec):
                raise TypeError('Value should be an instance of InputSpec. '
                                'Got: {}'.format(v))
        self._input_spec = value

    @property
    def non_trainable_variables(self):
        """Return the non-trainable variables.

        Alias for ``self.non_trainable_weights(...)``.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.Variable]
            The non-trainable variables.

        """
        return self.non_trainable_weights

    @property
    def trainable(self):
        """Return the trainable indicator.

        Returns
        -------
        bool
            ``True`` if trainable otherwise ``False``.

        """
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        """Set the trainable indicator.

        Parameters
        ----------
        value : bool
            The indicator.

        """
        for layer in self._flatten_layers():
            layer._trainable = value

    @property
    def trainable_weights(self):
        """Return the trainable weights.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.Variable]
            The trainable weights.

        """
        if self.trainable:
            nested = self._gather_children_attribute('trainable_variables')
            return self._dedupe_weights(self._trainable_weights + nested)
        return []

    @property
    def trainable_variables(self):
        """Return the trainable variables.

        Alias for ``self.trainable_weights(...)``.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.Variable]
            The trainable variables.

        """
        return self.trainable_weights

    @property
    def weights(self):
        """Return all the weights, both trainable and non-trainable.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.Variable]
            The weights.

        """
        return self.trainable_weights + self.non_trainable_weights

    def add_weight(
        self,
        name=None,
        shape=None,
        dtype=None,
        initializer=None,
        regularizer=None,
        trainable=True,
        use_resource=None,
        **kwargs
    ):
        """Add a new variable as the weight.

        Parameters
        ----------
        name : str, optional
            The optional variable name.
        shape : Sequence[int], optional
            The variable shape.
        dtype : str, optional
            The optional data type.
        initializer : Union[callable, str], optional
            The optional initializer.
        regularizer : Union[callable, str], optional
            The optional regularizer.
        trainable : bool, optional, default=True
            ``True`` to add to the ``trainable`` collection.
        use_resource : bool, optional, default=True
            ``True`` to set as a ``ResourceVariable``.

        """
        if shape is None:
            shape = ()
        initializer = initializers.get(initializer)
        regularizer = regularizers.get(regularizer)
        # Determine the data type.
        if dtype is None:
            dtype = self.dtype or dtypes.float32
        dtype = dtypes.as_dtype(dtype)
        # Determine the variable flags.
        trainable = True if trainable is None else trainable
        # Determine the initializer.
        if initializer is None:
            if dtype.is_floating:
                initializer = initializers.glorot_uniform()
            elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool:
                initializer = initializers.zeros()
            else:
                raise ValueError('Excepted an initializer for variable.')
        initial_value = initializer(shape, dtype=dtype)
        variable = Variable(initial_value, trainable=trainable)
        if regularizer is not None:
            variable = regularizer(variable)
        if trainable:
            self._trainable_weights.append(variable)
        else:
            self._non_trainable_weights.append(variable)
        return variable

    def build(self, input_shape):
        """Create and initialize the variables.

        Parameters
        ----------
        input_shape : Sequence[Sequence[int]]
            The input shape(s).

        """
        self.built = True

    def call(self, *args, **kwargs):
        """Define the implementation of forward.

        Parameters
        ----------
        args... : Sequence[dragon.Tensor]
            The input tensors.

        Returns
        -------
        Sequence[dragon.Tensor]
            The output tensors.

        """

    def load_weights(self, filepath, verbose=False):
        """Load the value of weights from a binary file.

        Parameters
        ----------
        filepath : str, required
            The path of weights file to load.
        verbose : bool, optional, default=False
            ``True`` to display the weights info.

        """
        if _is_hdf5_filepath(filepath):
            raise ValueError('HDF5 format will be supported in the future.')
        elif _is_pkl_filepath(filepath):
            with open(filepath, 'rb') as f:
                saving.load_weights_from_pickle(f, self, verbose)
        else:
            raise ValueError('TensorFlow format will never be supported.')

    def save_weights(self, filepath, save_format=None):
        """Save the value of weights into a binary file.

        Parameters
        ----------
        filepath : str, required
            The path of weights file to save.
        save_format : {'tf', 'h5', 'pkl'}, optional
            The determined saving format.

        """
        filepath_is_h5 = _is_hdf5_filepath(filepath)
        filepath_is_pkl = _is_pkl_filepath(filepath)
        if save_format is None:
            if filepath_is_h5:
                save_format = 'h5'
            elif filepath_is_pkl:
                save_format = 'pkl'
            else:
                save_format = 'tf'
        else:
            user_format = save_format.lower().strip()
            if user_format in ('tensorflow', 'tf'):
                save_format = 'tf'
            elif user_format in ('hdf5', 'h5', 'keras'):
                save_format = 'h5'
            else:
                raise ValueError('Unknown format "%s".\n'
                                 'Excepted format in (tf, h5, pkl).' % (save_format,))
        if save_format == 'tf':
            raise ValueError('TensorFlow format will never be supported.')
        if save_format == 'h5':
            raise ValueError('HDF5 format will be supported in the future.')
        with open(filepath, 'wb') as f:
            saving.save_weights_to_pickle(f, self)

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

    def _flatten_layers(self, recursive=True, include_self=True):
        """Flatten the layer instances."""
        for m in self._flatten_modules(recursive, include_self):
            if isinstance(m, Layer):
                yield m

    def _flatten_modules(self, recursive=True, include_self=True):
        """Flatten the module instances."""
        if include_self:
            yield self
        for m in self._flatten(recursive=recursive, predicate=module._is_module):
            yield m

    def _gather_children_attribute(self, attribute):
        """Return the attribute values of nested layers."""
        assert attribute in {'variables',
                             'trainable_variables',
                             'non_trainable_variables'}
        nested_layers = self._flatten_modules(recursive=False, include_self=False)
        values = (getattr(layer, attribute) for layer in nested_layers)
        return list(itertools.chain.from_iterable(values))

    def _maybe_build(self, inputs):
        if not self.built:
            input_spec.assert_input_compatibility(self.input_spec, inputs, self.name)
            inputs_list = nest.flatten(inputs)
            input_shapes = None
            if all(hasattr(x, 'shape') for x in inputs_list):
                input_shapes = [x.shape for x in inputs_list]
                if not nest.is_sequence(inputs):
                    input_shapes = input_shapes[0]
            self.build(input_shapes)

    def __call__(self, *args, **kwargs):
        """Wrap the ``self.call(...)`` with pre-post processing."""
        inputs = None
        if args:
            inputs, args = args[0], args[1:]
        with context.name_scope(self.name):
            self._maybe_build(inputs)
            outputs = self.call(inputs, *args, **kwargs)
        return outputs

    def __setattr__(self, name, value):
        # Add the layer.
        if isinstance(value, Layer):
            if value._name is None:
                value._name = name
        # Add the variables.
        for val in nest.flatten(value):
            if not isinstance(val, Variable):
                continue
            if val.trainable:
                if any(val is w for w in self._trainable_weights):
                    continue
                self._trainable_weights.append(val)
            else:
                if any(val is w for w in self._non_trainable_weights):
                    continue
                self._non_trainable_weights.append(val)
        # Add the attribute.
        object.__setattr__(self, name, value)


def _is_hdf5_filepath(filepath):
    """Predicate the filepath is a h5 file."""
    return (filepath.endswith('.h5') or filepath.endswith('.hdf5') or
            filepath.endswith('.keras'))


def _is_pkl_filepath(filepath):
    """Predicate the filepath is a pickle file."""
    return filepath.endswith('.pkl')
