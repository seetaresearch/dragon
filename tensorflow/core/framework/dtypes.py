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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/dtypes.py>
#
# ------------------------------------------------------------
"""Data type utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# Predefine the type enumerations
# to avoid to import the tensorflow proto
DT_INVALID = 0
DT_FLOAT = 1
DT_DOUBLE = 2
DT_INT32 = 3
DT_UINT8 = 4
DT_INT16 = 5
DT_INT8 = 6
DT_STRING = 7
DT_COMPLEX64 = 8
DT_INT64 = 9
DT_BOOL = 10
DT_QINT8 = 11
DT_QUINT8 = 12
DT_QINT32 = 13
DT_BFLOAT16 = 14
DT_QINT16 = 15
DT_QUINT16 = 16
DT_UINT16 = 17
DT_COMPLEX128 = 18
DT_HALF = 19
DT_VARIANT = 21
DT_UINT32 = 22
DT_UINT64 = 23

# Mappings between string names and enum values.
_STRING_TO_ENUM = {
    'float16': DT_HALF,
    'float32': DT_FLOAT,
    'float64': DT_DOUBLE,
    'int8': DT_INT8,
    'uint8': DT_UINT8,
    'int16': DT_INT16,
    'uint16': DT_UINT16,
    'int32': DT_INT32,
    'uint32': DT_UINT32,
    'int64': DT_INT64,
    'uint64': DT_UINT64,
    'string': DT_STRING,
    'complex64': DT_COMPLEX64,
    'complex128': DT_COMPLEX128,
    'bool': DT_BOOL,
    'qint8': DT_QINT8,
    'quint8': DT_QUINT8,
    'qint16': DT_QINT16,
    'quint16': DT_QUINT16,
    'qint32': DT_QINT32,
    'bfloat16': DT_BFLOAT16,
    'variant': DT_VARIANT,
}


class DType(str):
    """The basic data type.

    Following data types are defined:

    * ``tf.float16``: 16-bit half-precision floating-point.

    * ``tf.float32``: 32-bit single-precision floating-point.

    * ``tf.float64``: 64-bit double-precision floating-point.

    * ``tf.bfloat16``: 16-bit truncated floating-point.

    * ``tf.complex64``: 64-bit single-precision complex.

    * ``tf.complex128``: 128-bit double-precision complex.

    * ``tf.int8``: 8-bit signed integer.

    * ``tf.uint8``: 8-bit unsigned integer.

    * ``tf.uint16``: 16-bit unsigned integer.

    * ``tf.uint32``: 32-bit unsigned integer.

    * ``tf.uint64``: 64-bit unsigned integer.

    * ``tf.int16``: 16-bit signed integer.

    * ``tf.int32``: 32-bit signed integer.

    * ``tf.int64``: 64-bit signed integer.

    * ``tf.bool``: Boolean.

    * ``tf.string``: String.

    * ``tf.qint8``: Quantized 8-bit signed integer.

    * ``tf.quint8``: Quantized 8-bit unsigned integer.

    * ``tf.qint16``: Quantized 16-bit signed integer.

    * ``tf.quint16``: Quantized 16-bit unsigned integer.

    * ``tf.qint32``: Quantized 32-bit signed integer.

    * ``tf.variant``: Values of arbitrary types.

    """

    def __init__(self, value):
        """Create a ``DType``.

        Parameters
        ----------
        value : str
            The string name.

        """
        super(DType, self).__init__()
        self._name = value
        self._type_enum = _STRING_TO_ENUM[value]

    @property
    def as_numpy_dtype(self):
        """Return as the number of numpy data type.

        Returns
        -------
        type
            The data type number.

        """
        return _TF_TO_NP[self._type_enum]

    @property
    def as_datatype_enum(self):
        """Return as the enum value of data type.

        Returns
        -------
        int
            The enum value.

        """
        return self._type_enum

    @property
    def base_dtype(self):
        """Return the non-referenced data type.

        Returns
        -------
        dragon.vm.tensorflow.dtypes.DType
            The data type.

        """
        return self

    @property
    def is_numpy_compatible(self):
        """Return whether this data type is compatible with numpy.

        Returns
        -------
        bool
            ``True`` if compatible otherwise ``False``.

        """
        return self._type_enum in _TF_TO_NP

    @property
    def is_bool(self):
        """Return whether this is a boolean type.

        Returns
        -------
        bool
            ``True`` if this is a boolean type otherwise ``False``.

        """
        return self.base_dtype == bool

    @property
    def is_complex(self):
        """Return whether this is a complex type.

        Returns
        -------
        bool
            ``True`` if this is a complex type otherwise ``False``.

        """
        return self.base_dtype in (complex64, complex128)

    @property
    def is_floating(self):
        """Return whether this is a floating type.

        Returns
        -------
        bool
            ``True`` if this is a floating type otherwise ``False``.

        """
        return (self.is_numpy_compatible and
                issubclass(self.as_numpy_dtype, np.floating))

    @property
    def is_integer(self):
        """Return whether this is a integer type.

        Returns
        -------
        bool
            ``True`` if this is a integer type otherwise ``False``.

        """
        return (self.is_numpy_compatible and
                not self.is_quantized and
                issubclass(self.as_numpy_dtype, np.integer))

    @property
    def is_quantized(self):
        """Return whether this is a quantized type.

        Returns
        -------
        bool
            ``True`` if this is a quantized type otherwise ``False``.

        """
        return self.base_dtype in [qint8, quint8, qint16, quint16, qint32, bfloat16]

    @property
    def is_unsigned(self):
        """Return whether this is an unsigned type.

        Returns
        -------
        bool
            ``True`` if this is an unsigned type otherwise ``False``.

        """
        try:
            return self.min == 0
        except TypeError:
            return False

    @property
    def limits(self, clip_negative=True):
        """Return the numerical limits.

        Parameters
        ----------
        clip_negative : bool, optional, default=True
            ``True`` to return positive limits only.

        Returns
        -------
        Tuple[number, number]
            The limits.

        """
        min, max = dtype_range[self.as_numpy_dtype]
        if clip_negative:
            min = 0
        return min, max

    @property
    def max(self):
        """Return the max representable value.

        Returns
        -------
        number
            The max representable value.

        """
        if (self.is_quantized or self.base_dtype in
                (bool, string, complex64, complex128)):
            raise TypeError('Cannot find maximum value of %s.' % self)
        try:
            return np.finfo(self.as_numpy_dtype()).max
        except (TypeError, ValueError):
            try:
                return np.iinfo(self.as_numpy_dtype()).max
            except (TypeError, ValueError):
                raise TypeError('Cannot find maximum value of %s.' % self)

    @property
    def min(self):
        """Return the min representable value.

        Returns
        -------
        number
            The min representable value

        """
        if (self.is_quantized or self.base_dtype in
                (bool, string, complex64, complex128)):
            raise TypeError("Cannot find minimum value of %s." % self)
        try:
            return np.finfo(self.as_numpy_dtype()).min
        except (TypeError, ValueError):
            try:
                return np.iinfo(self.as_numpy_dtype()).min
            except (TypeError, ValueError):
                raise TypeError("Cannot find minimum value of %s." % self)

    @property
    def name(self):
        """Return the type name.

        Returns
        -------
        str
            The type name.

        """
        return self._name

    @property
    def real_dtype(self):
        """Return the data type of real part.

        Returns
        -------
        dragon.vm.tensorflow.dtypes.DType
            The data type of real part.

        """
        base = self.base_dtype
        if base == complex64:
            return float32
        elif base == complex128:
            return float64
        else:
            return self

    def is_compatible_with(self, other):
        """Return whether this data type can be converted as the other.

        Parameters
        ----------
        other : object
            The referring data type.

        Returns
        -------
        bool
            ``True`` if compatible otherwise ``False``.

        """
        other = as_dtype(other)
        return self._type_enum in (
            other.as_datatype_enum, other.base_dtype.as_datatype_enum)

    def __eq__(self, other):
        try:
            dtype = as_dtype(other).as_datatype_enum
            return self._type_enum == dtype
        except TypeError:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __int__(self):
        return self._type_enum

    def __str__(self):
        return self._name

    def __hash__(self):
        return self._type_enum


# Range of numpy dtype
dtype_range = {
    np.bool_: (False, True),
    np.bool8: (False, True),
    np.uint8: (0, 255),
    np.uint16: (0, 65535),
    np.int8: (-128, 127),
    np.int16: (-32768, 32767),
    np.int64: (-2 ** 63, 2 ** 63 - 1),
    np.uint64: (0, 2 ** 64 - 1),
    np.int32: (-2 ** 31, 2 ** 31 - 1),
    np.uint32: (0, 2 ** 32 - 1),
    np.float32: (-1, 1),
    np.float64: (-1, 1),
}

# Define standard wrappers for the string name.
float16 = half = DType('float16')
float32 = DType('float32')
float64 = double = DType('float64')
int32 = DType('int32')
uint8 = DType('uint8')
uint16 = DType('uint16')
uint64 = DType('uint32')
uint32 = DType('uint64')
int16 = DType('int16')
int8 = DType('int8')
string = DType('string')
complex64 = DType('complex64')
complex128 = DType('complex128')
int64 = DType('int64')
bool = DType('bool')
qint8 = DType('qint8')
quint8 = DType('quint8')
qint16 = DType('qint16')
quint16 = DType('quint16')
qint32 = DType('qint32')
bfloat16 = DType('bfloat16')
variant = DType('variant')

# Numpy representation for quantized dtypes.
_np_qint8 = np.dtype([("qint8", np.int8)])
_np_quint8 = np.dtype([("quint8", np.uint8)])
_np_qint16 = np.dtype([("qint16", np.int16)])
_np_quint16 = np.dtype([("quint16", np.uint16)])
_np_qint32 = np.dtype([("qint32", np.int32)])

_NP_TO_TF = {
    np.float16: float16,
    np.float32: float32,
    np.float64: float64,
    np.int32: int32,
    np.int64: int64,
    np.uint8: uint8,
    np.uint16: uint16,
    np.uint32: uint32,
    np.uint64: uint64,
    np.int16: int16,
    np.int8: int8,
    np.complex64: complex64,
    np.complex128: complex128,
    np.object_: string,
    np.string_: string,
    np.unicode_: string,
    np.bool_: bool,
    _np_qint8: qint8,
    _np_quint8: quint8,
    _np_qint16: qint16,
    _np_quint16: quint16,
    _np_qint32: qint32,
}

_TF_TO_NP = {
    DT_HALF: np.float16,
    DT_FLOAT: np.float32,
    DT_DOUBLE: np.float64,
    DT_INT32: np.int32,
    DT_UINT8: np.uint8,
    DT_UINT16: np.uint16,
    DT_INT16: np.int16,
    DT_INT8: np.int8,
    DT_STRING: np.object,
    DT_COMPLEX64: np.complex64,
    DT_COMPLEX128: np.complex128,
    DT_INT64: np.int64,
    DT_BOOL: np.bool,
    DT_QINT8: _np_qint8,
    DT_QUINT8: _np_quint8,
    DT_QINT16: _np_qint16,
    DT_QUINT16: _np_quint16,
    DT_QINT32: _np_qint32,
    DT_BFLOAT16: np.uint16,
}

_INTERN_TABLE = {
    DT_HALF: float16,
    DT_FLOAT: float32,
    DT_DOUBLE: float64,
    DT_INT32: int32,
    DT_UINT8: uint8,
    DT_UINT16: uint16,
    DT_UINT32: uint32,
    DT_UINT64: uint64,
    DT_INT16: int16,
    DT_INT8: int8,
    DT_STRING: string,
    DT_COMPLEX64: complex64,
    DT_COMPLEX128: complex128,
    DT_INT64: int64,
    DT_BOOL: bool,
    DT_QINT8: qint8,
    DT_QUINT8: quint8,
    DT_QINT16: qint16,
    DT_QUINT16: quint16,
    DT_QINT32: qint32,
    DT_BFLOAT16: bfloat16,
    DT_VARIANT: variant,
}

_STRING_TO_TF = {
    key: _INTERN_TABLE[value] for key, value in _STRING_TO_ENUM.items()
}

_ANY_TO_TF = {}
_ANY_TO_TF.update(_INTERN_TABLE)
_ANY_TO_TF.update(_STRING_TO_TF)
_ANY_TO_TF.update(_NP_TO_TF)


def as_dtype(type_value):
    """Convert a data type value into ``tf.DType``.

    Parameters
    ----------
    type_value : Any
        The data type.

    Returns
    -------
    dragon.vm.tensorflow.dtypes.DType
        The tensorflow data type.

    """
    if isinstance(type_value, DType):
        return type_value
    if isinstance(type_value, np.dtype):
        try:
            return _NP_TO_TF[type_value.type]
        except KeyError:
            pass
    try:
        return _ANY_TO_TF[type_value]
    except KeyError:
        pass
    raise TypeError("Cannot convert value %r to a TensorFlow DType." % type_value)
