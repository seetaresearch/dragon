# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

import numpy as np

from dragon.vm.tensorflow.protobuf import types_pb2


class DType(object):
    """
    The basic data type.
    """
    def __init__(self, type_enum):
        type_enum = int(type_enum)
        if (type_enum not in types_pb2.DataType.values()
            or type_enum == types_pb2.DT_INVALID):
            raise TypeError(
                "type_enum is not a valid types_pb2.DataType: %s" % type_enum)
        self._type_enum = type_enum

    @property
    def base_dtype(self):
        return self

    @property
    def real_dtype(self):
        base = self.base_dtype
        if base == complex64:
            return float32
        elif base == complex128:
            return float64
        else:
            return self

    @property
    def is_numpy_compatible(self):
        return (self._type_enum != types_pb2.DT_RESOURCE and
                self._type_enum != types_pb2.DT_RESOURCE_REF)

    @property
    def as_numpy_dtype(self):
        return _TF_TO_NP[self._type_enum]

    @property
    def as_datatype_enum(self):
        return self._type_enum

    @property
    def is_bool(self):
        return self.base_dtype == bool

    @property
    def is_integer(self):
        return (self.is_numpy_compatible and not self.is_quantized and
                issubclass(self.as_numpy_dtype, np.integer))

    @property
    def is_floating(self):
        return self.is_numpy_compatible and issubclass(self.as_numpy_dtype,
                                                       np.floating)

    @property
    def is_complex(self):
        return self.base_dtype in (complex64, complex128)

    @property
    def is_quantized(self):
        return self.base_dtype in [qint8, quint8, qint16, quint16, qint32, bfloat16]

    @property
    def is_unsigned(self):
        try:
            return self.min == 0
        except TypeError:
            return False

    @property
    def min(self):
        if (self.is_quantized or self.base_dtype in
            (bool, string, complex64, complex128)):
            raise TypeError("Cannot find minimum value of %s." % self)
        try:
            return np.finfo(self.as_numpy_dtype()).min
        except:
            try:
                return np.iinfo(self.as_numpy_dtype()).min
            except:
                raise TypeError("Cannot find minimum value of %s." % self)

    @property
    def max(self):
        if (self.is_quantized or self.base_dtype in
            (bool, string, complex64, complex128)):
            raise TypeError("Cannot find maximum value of %s." % self)
        try:
            return np.finfo(self.as_numpy_dtype()).max
        except:
            try:
                return np.iinfo(self.as_numpy_dtype()).max
            except:
                raise TypeError("Cannot find maximum value of %s." % self)

    @property
    def limits(self, clip_negative=True):
        min, max = dtype_range[self.as_numpy_dtype]
        if clip_negative:
            min = 0
        return min, max

    def is_compatible_with(self, other):
        other = as_dtype(other)
        return self._type_enum in (
            other.as_datatype_enum, other.base_dtype.as_datatype_enum)

    def __eq__(self, other):
        if other is None:
            return False
        try:
            dtype = as_dtype(other).as_datatype_enum
            return self._type_enum == dtype
        except TypeError:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def name(self):
        """Returns the string name for this `DType`."""
        return _TYPE_TO_STRING[self._type_enum]

    def __int__(self):
        return self._type_enum

    def __str__(self):
        return "<dtype: %r>" % self.name

    def __hash__(self):
        return self._type_enum


# Define data type range of numpy dtype
dtype_range = {np.bool_: (False, True),
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
               np.float64: (-1, 1)}


# Define standard wrappers for the types_pb2.DataType enum.
resource = DType(types_pb2.DT_RESOURCE)
float16 = DType(types_pb2.DT_HALF)
half = float16
float32 = DType(types_pb2.DT_FLOAT)
float64 = DType(types_pb2.DT_DOUBLE)
double = float64
int32 = DType(types_pb2.DT_INT32)
uint8 = DType(types_pb2.DT_UINT8)
uint16 = DType(types_pb2.DT_UINT16)
int16 = DType(types_pb2.DT_INT16)
int8 = DType(types_pb2.DT_INT8)
string = DType(types_pb2.DT_STRING)
complex64 = DType(types_pb2.DT_COMPLEX64)
complex128 = DType(types_pb2.DT_COMPLEX128)
int64 = DType(types_pb2.DT_INT64)
bool = DType(types_pb2.DT_BOOL)
qint8 = DType(types_pb2.DT_QINT8)
quint8 = DType(types_pb2.DT_QUINT8)
qint16 = DType(types_pb2.DT_QINT16)
quint16 = DType(types_pb2.DT_QUINT16)
qint32 = DType(types_pb2.DT_QINT32)
bfloat16 = DType(types_pb2.DT_BFLOAT16)


# Standard mappings between types_pb2.DataType values and string names.
_TYPE_TO_STRING = {
    types_pb2.DT_HALF: "float16",
    types_pb2.DT_FLOAT: "float32",
    types_pb2.DT_DOUBLE: "float64",
    types_pb2.DT_INT32: "int32",
    types_pb2.DT_UINT8: "uint8",
    types_pb2.DT_UINT16: "uint16",
    types_pb2.DT_INT16: "int16",
    types_pb2.DT_INT8: "int8",
    types_pb2.DT_STRING: "string",
    types_pb2.DT_COMPLEX64: "complex64",
    types_pb2.DT_COMPLEX128: "complex128",
    types_pb2.DT_INT64: "int64",
    types_pb2.DT_BOOL: "bool",
    types_pb2.DT_QINT8: "qint8",
    types_pb2.DT_QUINT8: "quint8",
    types_pb2.DT_QINT16: "qint16",
    types_pb2.DT_QUINT16: "quint16",
    types_pb2.DT_QINT32: "qint32",
    types_pb2.DT_BFLOAT16: "bfloat16",
    types_pb2.DT_RESOURCE: "resource",
    types_pb2.DT_HALF_REF: "float16_ref",
    types_pb2.DT_FLOAT_REF: "float32_ref",
    types_pb2.DT_DOUBLE_REF: "float64_ref",
    types_pb2.DT_INT32_REF: "int32_ref",
    types_pb2.DT_UINT8_REF: "uint8_ref",
    types_pb2.DT_UINT16_REF: "uint16_ref",
    types_pb2.DT_INT16_REF: "int16_ref",
    types_pb2.DT_INT8_REF: "int8_ref",
    types_pb2.DT_STRING_REF: "string_ref",
    types_pb2.DT_COMPLEX64_REF: "complex64_ref",
    types_pb2.DT_COMPLEX128_REF: "complex128_ref",
    types_pb2.DT_INT64_REF: "int64_ref",
    types_pb2.DT_BOOL_REF: "bool_ref",
    types_pb2.DT_QINT8_REF: "qint8_ref",
    types_pb2.DT_QUINT8_REF: "quint8_ref",
    types_pb2.DT_QINT16_REF: "qint16_ref",
    types_pb2.DT_QUINT16_REF: "quint16_ref",
    types_pb2.DT_QINT32_REF: "qint32_ref",
    types_pb2.DT_BFLOAT16_REF: "bfloat16_ref",
    types_pb2.DT_RESOURCE_REF: "resource_ref",
}

# Standard mappings between types_pb2.DataType values and numpy.dtypes.

# Numpy representation for quantized dtypes.
_np_qint8 = np.dtype([("qint8", np.int8, 1)])
_np_quint8 = np.dtype([("quint8", np.uint8, 1)])
_np_qint16 = np.dtype([("qint16", np.int16, 1)])
_np_quint16 = np.dtype([("quint16", np.uint16, 1)])
_np_qint32 = np.dtype([("qint32", np.int32, 1)])

_NP_TO_TF = frozenset([
    (np.float16, float16),
    (np.float32, float32),
    (np.float64, float64),
    (np.int32, int32),
    (np.int64, int64),
    (np.uint8, uint8),
    (np.uint16, uint16),
    (np.int16, int16),
    (np.int8, int8),
    (np.complex64, complex64),
    (np.complex128, complex128),
    (np.object, string),
    (np.bool, bool),
    (_np_qint8, qint8),
    (_np_quint8, quint8),
    (_np_qint16, qint16),
    (_np_quint16, quint16),
    (_np_qint32, qint32),
    # NOTE(touts): Intentionally no way to feed a DT_BFLOAT16.
])

_TF_TO_NP = {
    types_pb2.DT_HALF: np.float16,
    types_pb2.DT_FLOAT: np.float32,
    types_pb2.DT_DOUBLE: np.float64,
    types_pb2.DT_INT32: np.int32,
    types_pb2.DT_UINT8: np.uint8,
    types_pb2.DT_UINT16: np.uint16,
    types_pb2.DT_INT16: np.int16,
    types_pb2.DT_INT8: np.int8,
    # NOTE(touts): For strings we use np.object as it supports variable length
    # strings.
    types_pb2.DT_STRING: np.object,
    types_pb2.DT_COMPLEX64: np.complex64,
    types_pb2.DT_COMPLEX128: np.complex128,
    types_pb2.DT_INT64: np.int64,
    types_pb2.DT_BOOL: np.bool,
    types_pb2.DT_QINT8: _np_qint8,
    types_pb2.DT_QUINT8: _np_quint8,
    types_pb2.DT_QINT16: _np_qint16,
    types_pb2.DT_QUINT16: _np_quint16,
    types_pb2.DT_QINT32: _np_qint32,
    types_pb2.DT_BFLOAT16: np.uint16,
}


def as_dtype(type_value):
    if isinstance(type_value, DType):
        return type_value

    if isinstance(type_value, np.dtype):
        if type_value.type == np.string_ or type_value.type == np.unicode_:
            return string

    for key, val in _NP_TO_TF:
        try:
            if key == type_value:
                return val
        except TypeError as e:
            raise TypeError("Cannot convert {} to a dtype. {}".format(type_value, e))

    raise TypeError("Cannot convert value %r to a TensorFlow DType." % type_value)