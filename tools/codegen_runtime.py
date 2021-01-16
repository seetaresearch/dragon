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
"""Code generator for Runtime API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import shutil
import sys


class FileWriter(object):
    """Context-Manager to read and write content."""

    def __init__(self, f=None):
        self.f, self.content = f, None

    def __enter__(self):
        """Read and store the content."""
        with open(self.f, 'r', encoding='utf-8') as stream:
            self.content = stream.read()
        return self

    def __exit__(self, typ, value, traceback):
        """Exit and write the new content."""
        with open(self.f, 'w', encoding='utf-8') as stream:
            stream.write(self.content)
        self.content = None

    def apply_regex(self, files, transforms):
        for self.f in files:
            with self:
                for t in transforms:
                    self.content = re.sub(t, '', self.content)


def copy_dir(src, dest, enforce=True):
    """Copy the directory if necessary."""
    if os.path.exists(dest):
        if enforce:
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    else:
        shutil.copytree(src, dest)
    return dest


def glob_recurse(root_dir, *extensions):
    results = []
    for prefix, _, files in os.walk(root_dir):
        for file in files:
            name, extension = os.path.splitext(file)
            if extension in extensions:
                results.append(os.path.join(prefix, file))
    return results


def path_remove_gradient(project_source_dir):
    """Remove the codes of gradient utilities."""
    runtime_dir = project_source_dir + '/runtime'
    kernels_dir = project_source_dir + '/kernels'
    operators_dir = project_source_dir + '/operators'
    kernels_dir = copy_dir(kernels_dir, runtime_dir + '/kernels')
    operators_dir = copy_dir(operators_dir, runtime_dir + '/operators')
    FileWriter().apply_regex(
        glob_recurse(kernels_dir, '.cc', '.cu'), [
            r'DEFINE.*GRAD.*LAUNCHER.*[;]',
            r'DEFINE.*LAUNCHER.*Grad.*[;]',
        ]
    )
    FileWriter().apply_regex(
        glob_recurse(operators_dir, '.cc'), [
            r'DEPLOY_.+[(].*Gradient[)][;]',
            r'OPERATOR_SCHEMA[(].+Gradient.*[)][\s\S]*?[;]',
            r'REGISTER_GRADIENT[(].+[)][;]',
            r'class GradientMaker[\s\S]*[;]',
        ]
    )


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Usage: codegen.py '
                         '<PROJECT_SOURCE_DIR> <PATH_NAME>')
    project_source_dir, path_name = sys.argv[1:]

    if path_name == 'REMOVE_GRADIENT':
        path_remove_gradient(project_source_dir)
    else:
        raise ValueError('Unsupported path: ' + path_name)
