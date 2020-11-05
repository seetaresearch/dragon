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
#     <https://github.com/BVLC/caffe/blob/master/python/caffe/net_spec.py>
#
# ------------------------------------------------------------
"""Net proto maker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dragon.vm.caffe.core.proto import caffe_pb2


def param_name_dict():
    """Find out the correspondence between layer names and parameter names."""
    layer = caffe_pb2.LayerParameter()
    # Get all parameter names (typically underscore case) and corresponding
    # type names (typically camel case), which contain the layer names
    # (note that not all parameters correspond to layers, but we'll ignore that).
    param_names = [f.name for f in layer.DESCRIPTOR.fields if f.name.endswith('_param')]
    param_type_names = [type(getattr(layer, s)).__name__ for s in param_names]
    # Strip the final '_param' or 'Parameter'.
    param_names = [s[:-len('_param')] for s in param_names]
    param_type_names = [s[:-len('Parameter')] for s in param_type_names]
    return dict(zip(param_type_names, param_names))


def to_proto(*tops):
    """Generate a NetParameter that contains all layers needed to compute
    all arguments."""
    layers = collections.OrderedDict()
    autonames = collections.Counter()
    for top in tops:
        top.fn._to_proto(layers, {}, autonames)
    net = caffe_pb2.NetParameter()
    net.layer.extend(layers.values())
    return net


def assign_proto(proto, name, val):
    """Assign a Python object to a protobuf message, based on the Python
    type (in recursive fashion). Lists become repeated fields/messages, dicts
    become messages, and other types are assigned directly. For convenience,
    repeated fields whose values are not lists are converted to single-element
    lists; e.g., `my_repeated_int_field=3` is converted to
    `my_repeated_int_field=[3]`."""
    is_repeated_field = hasattr(getattr(proto, name), 'extend')
    if is_repeated_field and not isinstance(val, list):
        val = [val]
    if isinstance(val, list):
        if isinstance(val[0], dict):
            for item in val:
                proto_item = getattr(proto, name).add()
                for k, v in item.items():
                    assign_proto(proto_item, k, v)
        else:
            getattr(proto, name).extend(val)
    elif isinstance(val, dict):
        for k, v in val.items():
            assign_proto(getattr(proto, name), k, v)
    else:
        setattr(proto, name, val)


class Top(object):
    """A Top specifies a single output blob (which could be one of several
    produced by a layer.)"""

    def __init__(self, fn, n):
        """
        Initialize a function.

        Args:
            self: (todo): write your description
            fn: (int): write your description
            n: (int): write your description
        """
        self.fn = fn
        self.n = n

    def to_proto(self):
        """Generate a NetParameter that contains all layers needed to compute
        this top."""
        return to_proto(self)

    def _update(self, params):
        """
        Updates the parameters.

        Args:
            self: (todo): write your description
            params: (list): write your description
        """
        self.fn._update(params)

    def _to_proto(self, layers, names, autonames):
        """
        Convert a list of layers to a list of layers.

        Args:
            self: (todo): write your description
            layers: (todo): write your description
            names: (str): write your description
            autonames: (str): write your description
        """
        return self.fn._to_proto(layers, names, autonames)


class Function(object):
    """A Function specifies a layer, its parameters, and its inputs (which
    are Tops from other layers)."""

    def __init__(self, type_name, inputs, params):
        """
        Initialize inputs.

        Args:
            self: (todo): write your description
            type_name: (str): write your description
            inputs: (list): write your description
            params: (dict): write your description
        """
        self.type_name = type_name
        self.inputs = inputs
        self.params = params
        self.ntop = self.params.get('ntop', 1)
        # Use del to make sure kwargs are not double-processed as layer params.
        if 'ntop' in self.params:
            del self.params['ntop']
        self.in_place = self.params.get('in_place', False)
        if 'in_place' in self.params:
            del self.params['in_place']
        self.tops = tuple(Top(self, n) for n in range(self.ntop))

    def _get_name(self, names, autonames):
        """
        Get the name of a type.

        Args:
            self: (todo): write your description
            names: (str): write your description
            autonames: (str): write your description
        """
        if self not in names and self.ntop > 0:
            names[self] = self._get_top_name(self.tops[0], names, autonames)
        elif self not in names:
            autonames[self.type_name] += 1
            names[self] = self.type_name + str(autonames[self.type_name])
        return names[self]

    def _get_top_name(self, top, names, autonames):
        """
        Return the name of the top - type.

        Args:
            self: (todo): write your description
            top: (str): write your description
            names: (str): write your description
            autonames: (str): write your description
        """
        if top not in names:
            autonames[top.fn.type_name] += 1
            names[top] = top.fn.type_name + str(autonames[top.fn.type_name])
        return names[top]

    def _update(self, params):
        """
        Updates the params

        Args:
            self: (todo): write your description
            params: (list): write your description
        """
        self.params.update(params)

    def _to_proto(self, layers, names, autonames):
        """
        Helper function to protobian.

        Args:
            self: (todo): write your description
            layers: (todo): write your description
            names: (str): write your description
            autonames: (str): write your description
        """
        if self in layers:
            return
        bottom_names = []
        for inp in self.inputs:
            inp._to_proto(layers, names, autonames)
            bottom_names.append(layers[inp.fn].top[inp.n])
        layer = caffe_pb2.LayerParameter()
        layer.type = self.type_name
        layer.bottom.extend(bottom_names)

        if self.in_place:
            layer.top.extend(layer.bottom)
        else:
            for top in self.tops:
                layer.top.append(self._get_top_name(top, names, autonames))
        layer.name = self._get_name(names, autonames)

        for k, v in self.params.items():
            # special case to handle generic *params
            if k.endswith('param'):
                assign_proto(layer, k, v)
            else:
                try:
                    assign_proto(getattr(
                        layer, _param_names[self.type_name] + '_param'), k, v)
                except (AttributeError, KeyError):
                    assign_proto(layer, k, v)

        layers[self] = layer


class NetSpec(object):
    """A NetSpec contains a set of Tops (assigned directly as attributes).
    Calling NetSpec.to_proto generates a NetParameter containing all of the
    layers needed to produce all of the assigned Tops, using the assigned
    names."""

    def __init__(self):
        """
        Initialize this method

        Args:
            self: (todo): write your description
        """
        super(NetSpec, self).__setattr__('tops', collections.OrderedDict())

    def __setattr__(self, name, value):
        """
        Sets the value of an attribute.

        Args:
            self: (todo): write your description
            name: (str): write your description
            value: (todo): write your description
        """
        self.tops[name] = value

    def __getattr__(self, name):
        """
        Return the attribute of a given name.

        Args:
            self: (todo): write your description
            name: (str): write your description
        """
        return self.tops[name]

    def __setitem__(self, key, value):
        """
        Sets the value of a key.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (str): write your description
        """
        self.__setattr__(key, value)

    def __getitem__(self, item):
        """
        Return the value of an item.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        return self.__getattr__(item)

    def __delitem__(self, name):
        """
        Removes an item from the list.

        Args:
            self: (todo): write your description
            name: (str): write your description
        """
        del self.tops[name]

    def keys(self):
        """
        Return a copy of the keys.

        Args:
            self: (todo): write your description
        """
        keys = [k for k, v in self.tops.items()]
        return keys

    def vals(self):
        """
        Return a dict of all values in this collection.

        Args:
            self: (todo): write your description
        """
        vals = [v for k, v in self.tops.items()]
        return vals

    def update(self, name, params):
        """
        Update this parameter values.

        Args:
            self: (todo): write your description
            name: (str): write your description
            params: (list): write your description
        """
        self.tops[name]._update(params)

    def to_proto(self):
        """
        Convert the network to a protobuf.

        Args:
            self: (todo): write your description
        """
        names = {v: k for k, v in self.tops.items()}
        autonames = collections.Counter()
        layers = collections.OrderedDict()
        for name, top in self.tops.items():
            top._to_proto(layers, names, autonames)
        net = caffe_pb2.NetParameter()
        net.layer.extend(layers.values())
        return net


class Layers(object):
    """A Layers object is a pseudo-module which generates ops that specify
    layers; e.g., Layers().Convolution(bottom, kernel_size=3) will produce a Top
    specifying a 3x3 convolution applied to bottom."""

    def __getattr__(self, name):
        """
        Get the name of a given layer.

        Args:
            self: (todo): write your description
            name: (str): write your description
        """
        def layer_fn(*args, **kwargs):
            """
            Return the layer function.

            Args:
            """
            fn = Function(name, args, kwargs)
            if fn.ntop == 0:
                return fn
            elif fn.ntop == 1:
                return fn.tops[0]
            else:
                return fn.tops
        return layer_fn


class Parameters(object):
    """A Parameters object is a pseudo-module which generates constants used
    in layer parameters; e.g., Parameters().Pooling.MAX is the value used
    to specify max pooling."""

    def __getattr__(self, name):
        """
        Gets a single parameter.

        Args:
            self: (todo): write your description
            name: (str): write your description
        """
        class Param:
            def __getattr__(self, param_name):
                """
                Returns the value of a protobuf.

                Args:
                    self: (todo): write your description
                    param_name: (str): write your description
                """
                return getattr(getattr(caffe_pb2, name + 'Parameter'), param_name)
        return Param()


_param_names = param_name_dict()
layers = Layers()
params = Parameters()
