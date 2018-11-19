# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py>
#
# ------------------------------------------------------------

import warnings


class _Reduction:
    @staticmethod
    def get_enum(reduction):
        if reduction == 'none':
            return 0
        if reduction == 'elementwise_mean':
            return 1
        if reduction == 'sum':
            return 2
        raise ValueError(reduction + " is not a valid value for reduction")

    # In order to support previous versions, accept boolean size_average and reduce
    # and convert them into the new constants for now

    # We use these functions in torch/legacy as well, in which case we'll silence the warning
    @staticmethod
    def legacy_get_string(size_average, reduce, emit_warning=True):
        warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

        if size_average is None:
            size_average = True
        if reduce is None:
            reduce = True

        if size_average and reduce:
            ret = 'elementwise_mean'
        elif reduce:
            ret = 'sum'
        else:
            ret = 'none'
        if emit_warning:
            warnings.warn(warning.format(ret))
        return ret

    @staticmethod
    def legacy_get_enum(size_average, reduce, emit_warning=True):
        return _Reduction.get_enum(_Reduction.legacy_get_string(size_average, reduce, emit_warning))