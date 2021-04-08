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

import collections
import os

try:
    import h5py
except ImportError:
    h5py = None
import numpy

from dragon.core.util import nest
from dragon.core.util import six

PICKLE_DEFAULT_PROTOCOL = 2


def assign_weights(value_list, module):
    """Assign the value to the module weights.

    The number of values and weights should be equal,
    while weight shape could be wrong if order is not matched.

    Parameters
    ----------
    value_list : Sequence[array_like]
        The values to assign.
    module : dragon.vm.tensorlayer.Module
        The module to take weights.

    """
    matched_info = []
    weight_list = module.weights
    if len(weight_list) != len(value_list):
        raise ValueError(
            'Excepted %d values for weights, got %d.'
            % (len(weight_list), len(value_list)))
    for weight, value in zip(weight_list, value_list):
        _set_value(weight, value)
        matched_info.append((weight.name, weight.shape))
    return matched_info


def load_and_assign_npz(name, module):
    """Load and assign weights from a npz file.

    The number of values and weights should be equal,
    while weight shape could be wrong if order is not matched.

    Parameters
    ----------
    name : str
        The path of npz file.
    module : dragon.vm.tensorlayer.Module
        The module to take weights.

    """
    if not os.path.exists(name):
        raise ValueError("File {} doesn't exist.".format(name))
    weights = load_npz(name=name)
    return assign_weights(weights, module)


def load_and_assign_npz_dict(name, module, skip=False):
    """Load and assign weights from a npz file.

    Parameters
    ----------
    name : str
        The path of npz file.
    module : dragon.vm.tensorlayer.Module
        The module to take weights.
    skip : bool, optional, default=False
        ``True`` to skip the modules which is not found.

    """
    if not os.path.exists(name):
        raise ValueError("File {} doesn't exist.".format(name))
    value_dict = numpy.load(name, allow_pickle=True)
    weight_dict = {w.name: w for w in module.weights}
    return _assign_weights_from_dict(weight_dict, value_dict, skip=skip)


def load_and_assign_pkl_dict(name, module, skip=False):
    """Load and assign weights from a pkl file.

    Parameters
    ----------
    name : str
        The path of npz file.
    module : dragon.vm.tensorlayer.Module
        The module to take weights.
    skip : bool, optional, default=False
        ``True`` to skip the modules which is not found.

    """
    if not os.path.exists(name):
        raise ValueError("File {} doesn't exist.".format(name))

    try:
        with open(name, 'rb') as f:
            value_dict = six.moves.pickle.load(f)
    except UnicodeDecodeError:
        with open(name, 'rb') as f:
            value_dict = six.moves.pickle.load(f, encoding='bytes')

    weight_dict = {w.name: w for w in module.weights}
    return _assign_weights_from_dict(weight_dict, value_dict, skip=skip)


def load_hdf5_to_weights(filepath, module, skip=False):
    """Load weights by name from a hdf5 file.

    Parameters
    ----------
    filepath : str
        The path of weights file.
    module : dragon.vm.tensorlayer.Module
        The module to take weights.
    skip : bool, optional, default=False
        ``True`` to skip the modules which is not found.

    """
    f = h5py.File(filepath, 'r')
    try:
        _ = [n.decode('utf8') for n in f.attrs["layer_names"]]
    except Exception:
        raise NameError(
            "The loaded hdf5 file needs to have 'layer_names' as attributes.\n"
            "Please check whether this hdf5 file is saved from TL.")
    matched_info = _load_weights_from_hdf5_group(f, module.modules, skip)
    f.close()
    return matched_info


def load_npz(name):
    """Load a sequence of value from a npz file.

    Parameters
    ----------
    name : str
        The path of npz file.

    Returns
    -------
    Sequence[array_like]
        The value sequence.

    """
    return numpy.load(name, allow_pickle=True)['params']


def save_npz(save_list, name):
    """Save a sequence of value into a npz file.

    Parameters
    ----------
    save_list : Sequence[array_like]
        The values to be saved.
    name : str
        The path of saving file.

    """
    save_list = [] if save_list is None else save_list
    for i, input in enumerate(nest.flatten(save_list)):
        save_list[i] = _get_value(input)
    numpy.savez(name, params=save_list)


def save_npz_dict(save_list, name):
    """Save a sequence of value as dict into a pkl file.

    Parameters
    ----------
    save_list : Sequence[array_like]
        The values to be saved.
    name : str
        The path of saving file.

    """
    save_list = [] if save_list is None else save_list
    save_dict = collections.OrderedDict()
    for i, input in enumerate(nest.flatten(save_list)):
        if not hasattr(input, 'name'):
            raise ValueError('Input[%d] does not have <name> attribute.')
        save_dict[input.name] = _get_value(input)
    numpy.savez(name, **save_dict)


def save_pkl_dict(save_list, name):
    """Save a sequence of value as dict into a pkl file.

    Parameters
    ----------
    save_list : Sequence[array_like]
        The values to be saved.
    name : str
        The path of saving file.

    """
    save_list = [] if save_list is None else save_list
    save_dict = collections.OrderedDict()
    for i, input in enumerate(nest.flatten(save_list)):
        if not hasattr(input, 'name'):
            raise ValueError('Input[%d] does not have <name> attribute.')
        save_dict[input.name] = _get_value(input)
    with open(name, 'wb') as f:
        six.moves.pickle.dump(save_dict, f, PICKLE_DEFAULT_PROTOCOL)


def save_weights_to_hdf5(filepath, module):
    """Save weights into a hdf5 file.

    Parameters
    ----------
    filepath : str
        The path of weights file.
    module : dragon.vm.tensorlayer.Module
        The module to take weights.

    """
    if h5py is None:
        raise ImportError('Package <h5py> is required save weights.')
    with h5py.File(filepath, 'w') as f:
        _save_weights_to_hdf5_group(f, module.modules)


def _assign_weights_from_dict(weight_dict, value_dict, skip=False):
    """Assign the value to the module weights."""
    matched_info = []

    for key in value_dict.keys():
        if key not in weight_dict:
            if not skip:
                raise RuntimeError(
                    'Value <%s> not found in the weights.\n'
                    'Set <skip> to True to ignore it.' % key
                )
        else:
            value, weight = value_dict[key], weight_dict[key]
            # Set the value only if the number of elements is matched.
            if value.size != weight.size:
                raise ValueError(
                    'Weight <%s> requires %d elements, got %d.'
                    % (key, weight.size, value.size)
                )
            _set_value(weight, value)
            matched_info.append((key, weight.shape))

    return matched_info


def _get_value(input):
    """Return the value stolen from input."""
    if hasattr(input, 'numpy'):
        return input.numpy()
    return input


def _legacy_weights(module):
    """Return the weights stored in module self."""
    return module._trainable_weights + module._nontrainable_weights


def _load_weights_from_hdf5_group(f, modules, skip=False):
    """Load weights from a hdf5 group by name."""
    matched_info = []
    module_dict = {m.name: m for m in modules}
    module_names = [n.decode('utf8') for n in f.attrs["layer_names"]]
    for idx, name in enumerate(module_names):
        if name not in module_dict:
            if not skip:
                raise RuntimeError('Module <%s> not found.' % name)
        else:
            g = f[name]
            module = module_dict[name]
            weight_dict = {w.name: w for w in _legacy_weights(module)}
            value_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            value_dict = dict((name, g[name]) for name in value_names)
            matched_info += _assign_weights_from_dict(weight_dict, value_dict, skip=True)
    return matched_info


def _save_weights_to_hdf5_group(f, modules):
    """Save weights into hdf5 group recursively."""
    f.attrs['layer_names'] = [m.name.encode('utf8') for m in modules]
    for module in modules:
        g = f.create_group(module.name)
        weight_names, weight_values = [], []
        for w in _legacy_weights(module):
            value = _get_value(w)
            if value is not None:
                weight_names.append(w.name.encode('utf8'))
                weight_values.append(value)
        g.attrs['weight_names'] = weight_names
        for k, v in zip(weight_names, weight_values):
            val_dataset = g.create_dataset(k, v.shape, dtype=v.dtype)
            if not v.shape:
                val_dataset[()] = v
            else:
                val_dataset[:] = v


def _set_value(input, value):
    """Set the copied value to input."""
    if hasattr(input, '_impl'):
        input._impl.FromNumpy(value, True)
    else:
        raise ValueError('Input is not a legal tensor.')
