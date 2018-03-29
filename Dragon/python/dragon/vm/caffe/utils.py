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


def ToFillerArgs(FillerParamerer):
    kwargs = \
    {'value' : FillerParamerer.value,
     'low': FillerParamerer.min,
     'high': FillerParamerer.max,
     'mean': FillerParamerer.mean,
     'std': FillerParamerer.std}
    return kwargs

