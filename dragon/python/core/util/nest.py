# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Nest utility."""

import collections


def is_nested(input):
    """Return a bool indicating whether input is a sequence or dict.

    Parameters
    ----------
    input
        The input object.

    Returns
    -------
    bool
        ``True`` if input is a sequence or dict otherwise ``False``.

    """
    return is_sequence(input) or isinstance(input, dict)


def is_sequence(input):
    """Return a bool indicating whether input is a sequence.

    Parameters
    ----------
    input
        The input object.

    Returns
    -------
    bool
        ``True`` if input is a sequence otherwise ``False``.

    """
    return isinstance(input, collections.abc.Sequence) and not isinstance(input, str)


def flatten(input):
    """Return a flat list from the given input.

    Parameters
    ----------
    input
        The input object.

    Returns
    -------
    List
        The flat list of input.

    """

    def append_items(iterable, output_list):
        for item in iterable:
            if is_sequence(item):
                append_items(item, output_list)
            elif isinstance(item, dict):
                append_items(list(map(item.get, sorted(item.keys()))), output_list)
            else:
                output_list.append(item)

    output_list = []
    if is_sequence(input):
        append_items(input, output_list)
    elif isinstance(input, dict):
        append_items(list(map(input.get, sorted(input.keys()))), output_list)
    else:
        return [input]
    return output_list


def flatten_with_paths(input):
    """Return a flat list, yield as *(paths, element)*.

    Parameters
    ----------
    input
        The input object.

    Returns
    -------
    List[Tuple[Tuple, object]]
        The flat list of input.

    """
    return list(zip(yield_flatten_paths(input), flatten(input)))


def yield_flatten_paths(input):
    """Yield paths for nested structure.

    Parameters
    ----------
    input
        The input object.

    Returns
    -------
    Iterator[Tuple]
        The iterator of paths.

    """
    for k, _ in _yield_flatten_up_to(input, input, is_nested):
        yield k


def _yield_flatten_up_to(shallow_tree, input_tree, is_seq, path=()):
    """Return the tuple of path and element for iterable."""
    if not is_seq(shallow_tree):
        yield path, input_tree
    else:
        input_tree = dict(_yield_sorted_items(input_tree))
        for shallow_key, shallow_subtree in _yield_sorted_items(shallow_tree):
            sub_path = path + (shallow_key,)
            input_subtree = input_tree[shallow_key]
            for leaf_path, leaf_value in _yield_flatten_up_to(
                shallow_subtree, input_subtree, is_seq, path=sub_path
            ):
                yield leaf_path, leaf_value


def _yield_sorted_items(iterable):
    """Return the sorted iterable."""
    if isinstance(iterable, collections.abc.Mapping):
        for key in sorted(iterable):
            yield key, iterable[key]
    else:
        for item in enumerate(iterable):
            yield item
