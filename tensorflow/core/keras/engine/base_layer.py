# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/base_layer.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from dragon.core.framework import context
from dragon.core.util import nest
from dragon.vm.tensorflow.core.framework import dtypes
from dragon.vm.tensorflow.core.keras import initializers
from dragon.vm.tensorflow.core.keras import regularizers
from dragon.vm.tensorflow.core.keras import saving
from dragon.vm.tensorflow.core.keras.engine import input_spec
from dragon.vm.tensorflow.core.module import module
from dragon.vm.tensorflow.core.ops import variables as tf_variables
from dragon.vm.tensorflow.core.training.tracking import layer_utils


class Layer(module.Module):
    """The base class of layers.

    Inherit this class to design a new layer:

    ```python
    class MyLayer(tf.keras.layers.Layer):
        def __init__():
            super(MyModule, self).__init__()
    ```

    """

    def __init__(
        self,
        trainable=True,
        name=None,
        dtype=None,
        **kwargs
    ):
        """Create a ``Layer``.

        Parameters
        ----------
        trainable : bool, optional, default=True
            The initial training flag.
        name : str, optional
            The optional layer name.
        dtype : dragon.vm.tensorflow.dtypes.DType
            The optional data type.

        """
        super(Layer, self).__init__(name)
        self._trainable = trainable
        self._dtype = dtype

        # Modeling attributes.
        self.input_spec = None
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._layers = []

        # Indicating whether the variables are initialized.
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
    def layers(self):
        """Return the layers over all children.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.keras.layers.Layer]
            The sequence containing layers.

        """
        return layer_utils.filter_empty_layer_containers(self._layers)

    @property
    def non_trainable_weights(self):
        """Return the non-trainable weights.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.Variable]
            The sequence containing weights.

        """
        if self.trainable:
            nested = self._gather_children_attribute('non_trainable_weights')
            non_trainable_weights = self._non_trainable_weights + nested
        else:
            non_trainable_weights = (
                self._trainable_weights +
                self._non_trainable_weights +
                self._gather_children_attribute('weights')
            )
        return self._dedupe_weights(non_trainable_weights)

    @property
    def non_trainable_variables(self):
        """Return the non-trainable variables.

        Alias for ``self.non_trainable_weights(...)``.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.Variable]
            The sequence containing variables.

        """
        return self.non_trainable_weights

    @property
    def trainable(self):
        """Return the trainable indicator.

        Returns
        -------
        bool
            **True** if trainable otherwise **False**.

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
        self._trainable = value
        for layer in getattr(self, '_layers', []):
            layer.trainable = value

    @property
    def trainable_weights(self):
        """Return the trainable weights.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.Variable]
            The sequence containing weights.

        """
        if self.trainable:
            nested = self._gather_children_attribute('trainable_weights')
            return self._dedupe_weights(self._trainable_weights + nested)
        else:
            return []

    @property
    def trainable_variables(self):
        """Return the trainable variables.

        Alias for ``self.trainable_weights(...)``.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.Variable]
            The sequence containing variables.

        """
        return self.trainable_weights

    @property
    def weights(self):
        """Return all of the weights, both trainable and non-trainable.

        Returns
        -------
        Sequence[dragon.vm.tensorflow.Variable]
            The sequence containing weights.

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
            **True** to add to the ``trainable`` collection.
        use_resource : bool, optional, default=True
            **True** to set as a ``ResourceVariable``.

        """
        if shape is None:
            shape = ()
        initializer = initializers.get(initializer)
        regularizer = regularizers.get(regularizer)

        # Determine the data type
        if dtype is None:
            dtype = self.dtype or dtypes.float32
        dtype = dtypes.as_dtype(dtype)

        # Determine the variable flags
        trainable = True if trainable is None else trainable
        use_resource = True if use_resource is None else use_resource

        # Determine the initializer
        if initializer is None:
            if dtype.is_floating:
                initializer = initializers.glorot_uniform()
            elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool:
                initializer = initializers.zeros()
            else:
                raise ValueError('Excepted an initializer set for variable')

        variable = tf_variables.get_variable(
            name=name,
            shape=shape,
            initializer=initializer,
            regularizer=regularizer,
            dtype=dtype,
            trainable=trainable,
            use_resource=use_resource,
        )

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
            The input tensor(s).

        Returns
        -------
        Sequence[dragon.Tensor]
            The output tensor(s).

        """
        pass

    def load_weights(self, filepath, verbose=False):
        """Load the value of weights from a binary file.

        Parameters
        ----------
        filepath : str, required
            The path of weights file to load.
        verbose : bool, optional, default=False
            **True** to display the weights info.

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
                raise ValueError(
                    'Unknown format "%s".\n'
                    'Excepted format in (tf, h5, pkl).'
                    % (save_format,)
                )
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

    def _gather_children_attribute(self, attribute):
        """
        Gathers children of an attribute.

        Args:
            self: (todo): write your description
            attribute: (str): write your description
        """
        assert attribute in {
            'weights',
            'trainable_weights',
            'non_trainable_weights',
            'updates',
            'losses',
            'metrics',
        }
        if hasattr(self, '_layers'):
            layers = layer_utils \
                .filter_empty_layer_containers(self._layers)
            return list(
                itertools.chain.from_iterable(
                    getattr(layer, attribute) for layer in layers)
            )
        return []

    def _maybe_build(self, inputs):
        """
        Connects the graph to the graph.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        if not self.built:
            input_spec.assert_input_compatibility(
                self.input_spec, inputs, self.name)
            input_list = nest.flatten(inputs)
            input_shapes = None
            if all(hasattr(x, 'shape') for x in input_list):
                input_shapes = [x.shape for x in input_list]
                if not nest.is_sequence(inputs):
                    input_shapes = input_shapes[0]
            self.build(input_shapes)

    def _maybe_create_attribute(self, name, default_value):
        """Create the attribute if it hasn't been created."""
        if not hasattr(self, name):
            super(Layer, self).__setattr__(name, default_value)

    def _name_scope(self):
        """
        Returns the scope name.

        Args:
            self: (todo): write your description
        """
        return self.name

    def __call__(self, inputs, *args, **kwargs):
        """Wrap the ``self.call(...)`` with pre-post processing."""
        with context.name_scope(self._name_scope()):
            self._maybe_build(inputs)
            outputs = self.call(inputs, *args, **kwargs)
        return outputs

    def __setattr__(self, name, value):
        """
        Set weights.

        Args:
            self: (todo): write your description
            name: (str): write your description
            value: (todo): write your description
        """
        # Add the layer.
        if self.__class__.__name__ != 'Sequential' and \
                isinstance(value, Layer):
            if not any((layer is value for layer in self._layers)):
                if value._name is None:
                    value._name = name
                self._layers.append(value)
        # Add the variables.
        for val in nest.flatten(value):
            if not isinstance(val, tf_variables.VariableMetaclass):
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
    return (filepath.endswith('.h5') or
            filepath.endswith('.keras') or
            filepath.endswith('.hdf5'))


def _is_pkl_filepath(filepath):
    """Predicate the filepath is a pickle file."""
    return filepath.endswith('.pkl')
