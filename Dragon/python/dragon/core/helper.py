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

"""We really need some helpers at the frontend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy

from dragon.core import workspace as _workspace


class OperatorHelper(object):
    """A helper to infer the ``dtype`` and ``shape``.

    Achieve above exceptions in ``dragon.operators`` is trivial,
    so we move these dirty codes here.

    """

    _SIMPLE_APPLY = (
        # Following operators is the simplest case:
        # Input(0) => Output(0), shape and data type unchanged.
        'Relu', 'PRelu', 'Elu', 'SElu', 'Sigmoid', 'Tanh', 'Dropout', 'Softmax',
        'Add', 'Sub', 'Mul', 'Div', 'Clip', 'Log', 'Exp', 'Pow', 'Square', 'Sqrt',
        'Accumulate', 'Affine', 'Copy', 'Compare', 'StopGradient',  'MPIBroadcast',
        'BatchNorm', 'GroupNorm', 'L2Norm', 'LRN', 'BiasAdd', 'DropBlock2d',
    )

    @classmethod
    def get_index_and_name(cls, prefix='Op'):
        name = _workspace.GetDummyName(prefix, domain='Operator')
        try:
            _, op_idx = name.split('_')
        except:
            name = _workspace.GetDummyName(prefix, domain='Operator')
            _, op_idx = name.split('_')
        return int(op_idx), name

    @classmethod
    def get_name(cls, prefix='Op'):
        return cls.get_index_and_name(prefix)[1]

    @classmethod
    def apply(cls, op_type, arguments, inputs, outputs):
        method = '_apply_{}'.format(op_type)
        if hasattr(cls, method):
            return getattr(cls, method)(arguments, inputs, outputs)
        elif op_type in cls._SIMPLE_APPLY:
            return cls._apply_Simple(arguments, inputs, outputs)
        return outputs

    @classmethod
    def _apply_Simple(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        try: outputs[0].shape = inputs[0].shape[:]
        except: pass
        return outputs

    ###############################################
    #                                             #
    #                  Arithmetic                 #
    #                                             #
    ###############################################

    @classmethod
    def _apply_RAdd(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[1].dtype
        try:
            outputs[0].shape = inputs[1].shape[:]
        except:
            pass
        return outputs

    @classmethod
    def _apply_RSub(cls, arguments, inputs, outputs):
        return cls._apply_RAdd(arguments, inputs, outputs)

    @classmethod
    def _apply_RMul(cls, arguments, inputs, outputs):
        return cls._apply_RAdd(arguments, inputs, outputs)

    @classmethod
    def _apply_RDiv(cls, arguments, inputs, outputs):
        return cls._apply_RAdd(arguments, inputs, outputs)

    @classmethod
    def _apply_Maximum(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        if inputs[0].dtype is None:
            outputs[0].dtype = inputs[1].dtype
        try:
            outputs[0].shape = inputs[0].shape[:]
            if outputs[0].shape != inputs[1].shape and \
                len(outputs[0].shape) < len(inputs[1].shape):
                    outputs[0].shape = inputs[1].shape
        except:
            pass
        return outputs

    @classmethod
    def _apply_Minimum(cls, arguments, inputs, outputs):
        return cls._apply_Maximum(arguments, inputs, outputs)

    @classmethod
    def _apply_Moments(cls, arguments, inputs, outputs):
        outputs[0].dtype = outputs[1].dtype = \
            inputs[0].dtype if inputs[0].dtype == 'float64' else 'float32'
        axes, keep_dims = arguments['axes'], arguments['keep_dims']
        try:
            output_shape = inputs[0].shape[:]
            for axis in axes: output_shape[axis] = 1
            if not keep_dims:
                squeezed_shape = []
                for d in output_shape:
                    if d != 1: squeezed_shape.append(d)
                output_shape = squeezed_shape
        except:
            output_shape = None
        outputs[0].shape = outputs[1].shape = output_shape if axes else []
        return outputs

    @classmethod
    def _apply_Matmul(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        transA, transB = arguments['transA'], arguments['transB']
        try:
            outputs[0].shape = inputs[0].shape[:]
            try:
                M = inputs[0].shape[-1] if transA else inputs[0].shape[-2]
                outputs[0].shape[-2] = M
            except:
                pass
            try:
                N = inputs[1].shape[-2] if transB else inputs[1].shape[-1]
                outputs[0].shape[-1] = N
            except:
                pass
        except:
            pass
        return outputs

    @classmethod
    def _apply_Dot(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        transA, transB = arguments['transA'], arguments['transB']
        try:
            try:
                # Dot
                if len(inputs[0].shape) == 1:
                    outputs[0].shape = []
            except:
                pass
            try:
                # Matrix Multiplication (Right Alignment)
                if len(inputs[0].shape) >= 2 and \
                    len(inputs[1].shape) >= 2:
                    a_shape = inputs[0].shape[:] if not transA else inputs[0].shape[::-1]
                    b_shape = inputs[1].shape[:] if not transB else inputs[1].shape[::-1]
                    outputs[0].shape = [a_shape[0], None] if transA else a_shape
                    outputs[0].shape[-1] = b_shape[-1]
            except:
                pass
            try:
                # Matrix Vector Multiplication (Right Alignment)
                if len(inputs[0].shape) >= 2 and \
                    len(inputs[1].shape) == 1:
                    if transA:
                        outputs[0].shape = [inputs[0].shape[-1]]
                    else:
                        outputs[0].shape = inputs[0].shape[:]
                        del outputs[0].shape[-1]
            except:
                pass
        except:
            pass
        return outputs

    @classmethod
    def _apply_FullyConnected(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        axis, num_output = arguments['axis'], arguments['num_output']
        outputs[0].shape = [None] * (axis + 1)
        outputs[0].shape[axis] = num_output
        try: outputs[0].shape[:axis] = inputs[0].shape[:axis]
        except: pass
        return outputs

    @classmethod
    def _apply_Eltwise(cls, arguments, inputs, outputs):
        dtype, output_shape = None, None
        for input in inputs:
            if dtype is None: dtype = input.dtype
            if output_shape is None and input.shape is not None:
                output_shape = input.shape[:]
        for output in outputs:
            output.dtype = dtype
            output.shape = output_shape
        return outputs

    @classmethod
    def _apply_GramMatrix(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        outputs[0].shape = [None] * 3
        axis = arguments['axis']
        for i in range(3):
            try:
                if i == 0:
                    outputs[0].shape[i] = numpy.prod(inputs[0].shape[:axis])
                if i >= 1:
                    outputs[0].shape[i] = inputs[0].shape[axis]
            except: pass
        return outputs

    ###############################################
    #                                             #
    #                    Data                     #
    #                                             #
    ###############################################

    @classmethod
    def _apply_ImageData(cls, arguments, inputs, outputs):
        outputs[0].dtype = arguments['dtype']
        data_format = arguments['data_format']
        try:
            outputs[0].shape = inputs[0].shape[:]
            if data_format == 'NCHW':
                outputs[0].shape[1:4] = \
                    inputs[0].shape[3], \
                        inputs[0].shape[1], \
                            inputs[0].shape[2]
        except:
            pass
        return outputs

    ###############################################
    #                                             #
    #                 Initializer                 #
    #                                             #
    ###############################################

    @classmethod
    def _apply_Fill(cls, arguments, inputs, outputs):
        try: outputs[0].dtype = arguments['dtype']
        except: outputs[0].dtype = 'float32'
        dims = arguments['dims']
        try: outputs[0].shape = dims[:]
        except: pass
        return outputs

    @classmethod
    def _apply_RandomUniform(cls, arguments, inputs, outputs):
        return cls._apply_Fill(arguments, inputs, outputs)

    @classmethod
    def _apply_RandomNormal(cls, arguments, inputs, outputs):
        return cls._apply_Fill(arguments, inputs, outputs)

    @classmethod
    def _apply_TruncatedNormal(cls, arguments, inputs, outputs):
        return cls._apply_Fill(arguments, inputs, outputs)

    @classmethod
    def _apply_GlorotUniform(cls, arguments, inputs, outputs):
        return cls._apply_Fill(arguments, inputs, outputs)

    @classmethod
    def _apply_GlorotNormal(cls, arguments, inputs, outputs):
        return cls._apply_Fill(arguments, inputs, outputs)

    ###############################################
    #                                             #
    #                    Loss                     #
    #                                             #
    ###############################################

    @classmethod
    def _apply_NLLLoss(cls, arguments, inputs, outputs):
        outputs[0].dtype = 'float32'
        axis = arguments['axis']
        normalization = arguments['normalization']
        if normalization != 'UNIT': outputs[0].shape = []
        else:
            try:
                outputs[0].shape = inputs[0].shape[:]
                outputs[0].shape.pop(axis)
            except:
                pass
        return outputs

    @classmethod
    def _apply_SparseSoftmaxCrossEntropy(cls, arguments, inputs, outputs):
        return cls._apply_NLLLoss(arguments, inputs, outputs)

    @classmethod
    def _apply_SigmoidCrossEntropy(cls, arguments, inputs, outputs):
        outputs[0].dtype = 'float32'
        normalization = arguments['normalization']
        if normalization != 'UNIT': outputs[0].shape = []
        else:
            try:
                outputs[0].shape = inputs[0].shape[:]
            except:
                pass
        return outputs

    @classmethod
    def _apply_SoftmaxCrossEntropy(cls, arguments, inputs, outputs):
        return cls._apply_NLLLoss(arguments, inputs, outputs)

    @classmethod
    def _apply_SmoothL1Loss(cls, arguments, inputs, outputs):
        outputs[0].dtype, outputs[0].shape = 'float32', []
        return outputs

    @classmethod
    def _apply_L1Loss(cls, arguments, inputs, outputs):
        return cls._apply_SmoothL1Loss(arguments, inputs, outputs)

    @classmethod
    def _apply_L2Loss(cls, arguments, inputs, outputs):
        return cls._apply_SmoothL1Loss(arguments, inputs, outputs)

    @classmethod
    def _apply_SigmoidFocalLoss(cls, arguments, inputs, outputs):
        return cls._apply_NLLLoss(arguments, inputs, outputs)

    @classmethod
    def _apply_SoftmaxFocalLoss(cls, arguments, inputs, outputs):
        return cls._apply_NLLLoss(arguments, inputs, outputs)

    @classmethod
    def _apply_CTCLoss(cls, arguments, inputs, outputs):
        return cls._apply_SmoothL1Loss(arguments, inputs, outputs)

    ###############################################
    #                                             #
    #                    Misc                     #
    #                                             #
    ###############################################

    @classmethod
    def _apply_AsType(cls, arguments, inputs, outputs):
        outputs[0].dtype = arguments['dtype']
        try:
            outputs[0].shape = inputs[0].shape
        except:
            pass
        return outputs

    @classmethod
    def _apply_Accuracy(cls, arguments, inputs, outputs):
        outputs[0].dtype = 'float32'
        outputs[0].shape = []
        return outputs

    ###############################################
    #                                             #
    #                    MPI                      #
    #                                             #
    ###############################################

    @classmethod
    def _apply_MPIGather(cls, arguments, inputs, outputs):
        for i in range(len(outputs)):
            outputs[i].dtype = inputs[0].dtype
        try:
            if isinstance(outputs, list):
                for output in outputs:
                    output.shape = inputs[0].shape[:]
        except:
            pass
        return outputs

    ###############################################
    #                                             #
    #                    Array                    #
    #                                             #
    ###############################################

    @classmethod
    def _apply_Gather(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        axis = arguments['axis']
        try:
            outputs[0].shape = \
                inputs[0].shape[:axis] + \
                    inputs[1].shape[:] + \
                        inputs[0].shape[axis + 1:]
        except:
            pass
        return outputs

    @classmethod
    def _apply_RandomPick(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        outputs[1].dtype = 'int32'
        max_samples = arguments['max_samples']
        try:
            outputs[0].shape = inputs[0].shape[:]
            outputs[0].shape[arguments['axis']] = max_samples
        except:
            pass
        outputs[1].shape = [max_samples]
        return outputs

    @classmethod
    def _apply_Crop(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        return outputs

    @classmethod
    def _apply_Slice(cls, arguments, inputs, outputs):
        num_outputs = len(outputs)
        for i in range(num_outputs):
            outputs[i].dtype = inputs[0].dtype
        axis = arguments['axis']
        slice_points = arguments['slice_points']
        if slice_points is not None and len(slice_points) == 0:
            slice_points = None
        slice_offset = 0
        for i in range(len(outputs)):
            try:
                outputs[i].shape = inputs[0].shape[:]
            except:
                pass
            if slice_points is not None:
                slice_dim = None
                try:
                    if i < len(outputs) - 1:
                        slice_dim = slice_points[i] - slice_offset
                        slice_offset += slice_dim
                    else:
                        slice_dim = inputs[0].shape[axis] - slice_offset
                except:
                    pass
                outputs[i].shape[axis] = slice_dim
            else:
                try:
                    outputs[i].shape[axis] = \
                        outputs[i].shape[axis] // num_outputs
                except:
                    pass
        return outputs

    @classmethod
    def _apply_Stack(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        axis = arguments['axis']
        # Try to restore the number of dimensions
        for input in inputs:
            if outputs[0].shape is None and \
                    input.shape is not None:
                outputs[0].shape = input.shape[:]
        try:
            # Try to restore the dimensions
            for i in range(len(outputs[0].shape)):
                for input in inputs:
                    try:
                        if input.shape[i] is not None:
                            outputs[0].shape[i] = input.shape[i]
                    except:
                        pass
        except:
            pass
        try:
            axis += (len(outputs[0].shape) + 1)
            if axis < 0 or axis >= len(inputs[0].shape):
                outputs[0].shape.append(len(inputs))
            else:
                outputs[0].shape.insert(axis, len(inputs))
        except:
            pass
        return outputs

    @classmethod
    def _apply_Concat(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        axis = arguments['axis']
        # Try to restore the number of dimensions
        for input in inputs:
            if outputs[0].shape is None and \
                    input.shape is not None:
                outputs[0].shape = input.shape[:]
        try:
            # Try to restore the dimensions
            for i in range(len(outputs[0].shape)):
                if i == axis: continue
                for input in inputs:
                    try:
                        if input.shape[i] is not None:
                            outputs[0].shape[i] = input.shape[i]
                    except:
                        pass
        except:
            pass
        try:
            # Try to restore the concat dimension
            outputs[0].shape[axis], concat_dim = None, 0
            for input in inputs:
                concat_dim += input.shape[axis]
            outputs[0].shape[axis] = concat_dim
        except:
            pass
        return outputs

    @classmethod
    def _apply_Reduce(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        axes, keep_dims = arguments['axes'], arguments['keep_dims']
        try:
            output_shape = inputs[0].shape[:]
            for axis in axes: output_shape[axis] = 1
            if not keep_dims:
                squeezed_shape = []
                for d in output_shape:
                    if d != 1: squeezed_shape.append(d)
                output_shape = squeezed_shape
        except:
            output_shape = None
        outputs[0].shape = output_shape if axes else []
        return outputs

    @classmethod
    def _apply_ArgReduce(cls, arguments, inputs, outputs):
        outputs[0].dtype = 'int64'
        axis, top_k = arguments['axis'], arguments['top_k']
        try:
            outputs[0].shape = inputs[0].shape[:]
            if arguments['keep_dims']:
                if axis == -1: outputs[0].shape = [1]
                else: outputs[0].shape[axis] = top_k
            else:
                if axis == -1:
                    if top_k > 1: outputs[0].shape = [top_k]
                    else: outputs[0].shape = []
                else:
                    if top_k > 1: outputs[0].shape[axis] = top_k
                    else: del outputs[0].shape[axis]
        except:
            pass
        if len(outputs) == 2:
            outputs[1].dtype = inputs[0].dtype
            try:
                outputs[1].shape = outputs[0].shape[:]
            except:
                pass
        return outputs

    @classmethod
    def _apply_Transpose(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        perm = arguments['perm']
        try:
            # Missing perm if perm is dynamic
            if len(perm) == 0:
                perm = list(range(((len(inputs[0].shape)) - 1), -1, -1))
            outputs[0].shape = inputs[0].shape[:]
            for i, axis in enumerate(perm):
                outputs[0].shape[i] = inputs[0].shape[axis]
        except:
            pass
        return outputs

    @classmethod
    def _apply_Repeat(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        axis, repeats = arguments['axis'], arguments['repeats']
        try:
            # Repeat to a vector
            if axis is None:
                try:
                    fake_shape = inputs[0].shape[:]
                    total_count = numpy.prod(fake_shape)
                    outputs[0].shape = [total_count * repeats]
                except:
                    outputs[0].shape = [None]
            else:
                outputs[0].shape = inputs[0].shape[:]
                try:
                    outputs[0].shape[axis] *= repeats
                except:
                    outputs[0].shape[axis] = None
        except:
            pass
        return outputs

    @classmethod
    def _apply_OneHot(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        try:
            outputs[0].shape = inputs[0].shape[:]
            outputs[0].shape.append(arguments['depth'])
        except:
            pass
        return outputs

    @classmethod
    def _apply_Tile(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        multiples = arguments['multiples']
        try:
            outputs[0].shape = inputs[0].shape[:]
            for i, multiple in enumerate(multiples):
                try:
                    outputs[0].shape[i] *= multiple
                except:
                    pass
        except:
            pass
        return outputs

    @classmethod
    def _apply_Pad(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        pad_l, pad_r = arguments['pad_l'], arguments['pad_r']
        try:
            outputs[0].shape = inputs[0].shape[:]
            for i in range(len(pad_l)):
                try:
                    outputs[0].shape[i] += (pad_l[i] + pad_r[i])
                except:
                    pass
        except:
            pass
        return outputs

    @classmethod
    def _apply_Flatten(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        keep_axes = arguments['keep_axes']
        axis, num_axes = arguments['axis'], arguments['num_axes']
        try:
            fake_shape = inputs[0].shape[:]
            fake_shape = [1 if dim is None else dim for dim in fake_shape]
            if keep_axes is not None:
                keep_axes = min(keep_axes, len(inputs.shape))
                total_count = numpy.prod(fake_shape)
                outputs[0].shape = []
                for i in range(keep_axes - 1):
                    outputs[0].shape.append(inputs[0].shape[i])
                    total_count *= fake_shape[i]
                if total_count != 1:
                    outputs[0].shape.append(total_count)
            else:
                if num_axes == -1:
                    num_axes = len(inputs[0].shape) - axis
                num_axes = max(num_axes, 1)
                num_flatten = numpy.prod(fake_shape[axis : axis + num_axes])
                outputs[0].shape = \
                    inputs[0].shape[: axis] + [num_flatten] \
                        + inputs[0].shape[axis + num_axes:]
        except:
            pass
        return outputs

    @classmethod
    def _apply_Reshape(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        shape = arguments['dims']
        try:
            outputs[0].shape = [None] * len(shape)
            n_elements, n_elements_known = None, None
            try:
                n_elements = int(numpy.prod(inputs[0].shape))
            except:
                pass
            for i, s in enumerate(shape):
                try:
                    if s == -1: outputs[0].shape[i] = 1
                    elif s == 0: outputs[0].shape[i] = inputs[0].shape[i]
                    else: outputs[0].shape[i] = s
                except:
                    pass
            try:
                n_elements_known = int(numpy.prod(outputs[0].shape))
            except:
                pass
            for i, s in enumerate(shape):
                if s == -1:
                    try:
                        outputs[0].shape[i] = n_elements // n_elements_known
                    except:
                        outputs[0].shape[i] = None
        except:
            pass
        return outputs

    @classmethod
    def _apply_Squeeze(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        axis = arguments['axis']
        try:
            output_shape = []
            for idx, dim in enumerate(inputs[0].shape[:]):
                try:
                    if dim != 1:
                        output_shape.append(dim)
                    else:
                        if axis is not None and axis != idx:
                            output_shape.append(dim)
                except:
                    pass
            outputs[0].shape = output_shape
        except:
            pass
        return outputs

    @classmethod
    def _apply_ExpandDims(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        axis = arguments['axis']
        try:
            outputs[0].shape = inputs[0].shape[:]
            axis += (0 if axis >= 0 else len(inputs[0].shape) + 1)
            if axis < 0 or axis >= len(inputs[0].shape):
                outputs[0].shape.append(1)
            else:
                outputs[0].shape.insert(axis, 1)
        except:
            pass
        return outputs

    @classmethod
    def _apply_Shape(cls, arguments, inputs, outputs):
        outputs[0].dtype = 'int64'
        try:
            outputs[0].shape = [len(inputs[0].shape)]
        except:
            pass
        return outputs

    @classmethod
    def _apply_Arange(cls, arguments, inputs, outputs):
        outputs[0].dtype = arguments['dtype']
        count = None
        start, stop, step = \
            arguments['start'], \
                arguments['stop'], \
                    arguments['step']
        try:
            if stop is None:
                if start is not None and start != 0:
                    stop = start
                    start = 0
                    try:
                        count = int((stop - start - 1) / step) + 1
                    except:
                        pass
            else:
                try:
                    count = int((stop - start - 1) / step) + 1
                except:
                    pass
        except:
            pass
        outputs[0].shape = [count]
        return outputs

    @classmethod
    def _apply_Multinomial(cls, arguments, inputs, outputs):
        outputs[0].dtype = 'int64'
        try:
            outputs[0].shape = inputs[0].shape[:]
            outputs[0].shape[-1] = arguments['num_samples']
        except:
            pass
        return outputs

    ###############################################
    #                                             #
    #                    Vision                   #
    #                                             #
    ###############################################

    @classmethod
    def _apply_Conv2d(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        try:
            outputs[0].shape = inputs[0].shape[:]
            channel_axis = 1 if arguments['data_format'] == 'NCHW' else -1
            spatial_axis = 2 if arguments['data_format'] == 'NCHW' else 1
            outputs[0].shape[channel_axis] = arguments['num_output']
            for i in range(2):
                input_size = outputs[0].shape[i + spatial_axis]
                k = arguments['kernel_shape'][i]
                s = arguments['strides'][i]
                pl, pr = arguments['pads'][i], arguments['pads'][i + 2]
                dk, dp = (k - 1) + 1, pl + pr
                if 'SAME' not in arguments['padding']:
                    # Explicit pads
                    outputs[0].shape[i + spatial_axis] = int(float(input_size + dp - dk) / s) + 1
                else:
                    # Auto pads
                    outputs[0].shape[i + spatial_axis] = int(float(input_size + s - 1) / s)
        except:
            pass
        return outputs

    @classmethod
    def _apply_ConvTranspose2d(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        try:
            outputs[0].shape = inputs[0].shape[:]
            channel_axis = 1 if arguments['data_format'] == 'NCHW' else -1
            spatial_axis = 2 if arguments['data_format'] == 'NCHW' else 1
            outputs[0].shape[channel_axis] = arguments['num_output']
            for i in range(2):
                k = arguments['kernel_shape'][i]
                s = arguments['strides'][i]
                d = arguments['dilations'][i]
                pl, pr = arguments['pads'][i], arguments['pads'][i + 2]
                dk, dp = d * (k - 1) + 1, pl + pr
                input_size = outputs[0].shape[i + spatial_axis]
                if 'SAME' not in arguments['padding']:
                    # Explicit pads
                    outputs[0].shape[i + spatial_axis] = s * (input_size - 1) + dk - dp
                else:
                    # Auto pads
                    output_padding = arguments['output_padding']
                    output_shape = arguments['output_shape']
                    try:
                        outputs[0].shape[i + spatial_axis] = \
                            s * (input_size - 1) + dk + output_padding[i]
                        continue # Ignore the output shape
                    except:
                        outputs[0].shape[i + spatial_axis] = None
                    try:
                        outputs[0].shape[i + spatial_axis] = output_shape[i]
                    except:
                        outputs[0].shape[i + spatial_axis] = None
        except:
            pass
        return outputs

    @classmethod
    def _apply_DepthwiseConv2d(cls, arguments, inputs, outputs):
        return cls._apply_Conv2d(arguments, inputs, outputs)

    @classmethod
    def _apply_Pool2d(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        try:
            outputs[0].shape = inputs[0].shape[:]
            spatial_axis = 2 if arguments['data_format'] == 'NCHW' else 1
            for i in range(2):
                k = arguments['kernel_shape'][i]
                s = arguments['strides'][i]
                pl, pr = arguments['pads'][i], arguments['pads'][i + 2]
                if not arguments['global_pooling']:
                    if 'SAME' not in arguments['padding']:
                        # Explicit pads
                        input_size = outputs[0].shape[i + spatial_axis]
                        floor_or_ceil = math.ceil if arguments['ceil'] else math.floor
                        output_size = int(floor_or_ceil(float(
                            outputs[0].shape[i + spatial_axis] + pl + pr - k) / s) + 1)
                        if ((output_size - 1) * s >= (input_size + pl + pr)):
                            output_size = output_size - 1
                        outputs[0].shape[i + spatial_axis] = output_size
                    else:
                        # Auto pads
                        outputs[0].shape[i + spatial_axis] = \
                            int(math.ceil(float(outputs[0].shape[i + spatial_axis]) / s))
                else:
                    outputs[0].shape[i + spatial_axis] = 1
        except:
            pass
        return outputs

    @classmethod
    def _apply_ROIPool(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        pool_h, pool_w = arguments['pool_h'], arguments['pool_w']
        try:
            outputs[0].shape = inputs[0].shape[:]
            outputs[0].shape[2:4] = pool_h, pool_w
            try:
                outputs[0].shape[0] = inputs[1].shape[0]
            except:
                outputs[0].shape[0] = None
        except:
            pass
        return outputs

    @classmethod
    def _apply_ROIAlign(cls, arguments, inputs, outputs):
        return cls._apply_ROIPool(arguments, inputs, outputs)

    @classmethod
    def _apply_NNResize(cls, arguments, inputs, outputs):
        outputs[0].dtype = inputs[0].dtype
        try:
            outputs[0].shape = inputs[0].shape[:]
            spatial_axis = 2 if arguments['data_format'] == 'NCHW' else 1
            for i in range(2):
                output_dim = None
                try:
                    output_dim = arguments['dsize'][i]
                except:
                    try:
                        output_dim = int(float(outputs[0].shape[spatial_axis + i])
                            * ([arguments['fy'], arguments['fx']])[i])
                    except:
                        pass
                try:
                    outputs[0].shape[spatial_axis + i] = output_dim
                except:
                    pass
        except:
            pass
        return outputs

    @classmethod
    def _apply_BilinearResize(cls, arguments, inputs, outputs):
        return cls._apply_NNResize(arguments, inputs, outputs)


class GradientHelper(object):
    """A helper to store the known gradient relations.

    Each ``Tensor`` will hold this helper.

    """
    def __init__(self, parent):
        self._parent = parent
        self._cost, self._wrt = [], []

    def add_cost(self, cost):
        self._cost.append(cost)

    def add_wrt(self, wrt):
        self._wrt.append(wrt)

    def make_pairs(self):
        return [(self._parent.name, wrt) for wrt in self._wrt]

    def required(self):
        return len(self._wrt) > 0

    @property
    def cost(self):
        return self._cost

    @property
    def wrt(self):
        return self._wrt