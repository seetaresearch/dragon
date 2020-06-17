# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/tracking/layer_utils.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def filter_empty_layer_containers(layer_list):
    """Filter out nested layers from containers."""
    existing = set()
    to_visit = layer_list[::-1]
    filtered = []
    while to_visit:
        obj = to_visit.pop()
        obj_id = id(obj)
        if obj_id in existing:
            continue
        existing.add(obj_id)
        if hasattr(obj, '_layers'):
            filtered.append(obj)
            to_visit.extend(obj.layers[::-1])
    return filtered
