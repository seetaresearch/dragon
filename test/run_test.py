# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
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

import sys
import subprocess

import argparse

TESTS_AND_SOURCES = [
    ('dragon/test_autograph', 'dragon.core'),
    ('dragon/test_device', 'dragon.core'),
    ('dragon/test_distributed', 'dragon.core'),
    ('dragon/test_framework', 'dragon.core'),
    ('dragon/test_io', 'dragon.core'),
    ('dragon/test_ops', 'dragon.core'),
    ('dragon/test_util', 'dragon.core'),
    ('torch/test_autograd', 'dragon.vm.torch.core'),
    ('torch/test_jit', 'dragon.vm.torch.core'),
    ('torch/test_nn', 'dragon.vm.torch.core'),
    ('torch/test_ops', 'dragon.vm.torch.core'),
    ('torch/test_optim', 'dragon.vm.torch.core'),
    ('torch/test_torch', 'dragon.vm.torch.core'),
]

TESTS = [t[0] for t in TESTS_AND_SOURCES]
SOURCES = [t[1] for t in TESTS_AND_SOURCES]


def parse_args():
    parser = argparse.ArgumentParser(
        description='run the unittests',
        epilog='where TESTS is any of: {}'.format(', '.join(TESTS)))
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='print verbose information')
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help='print error information only')
    parser.add_argument(
        '-c',
        '--coverage',
        action='store_true',
        help='run coverage for unittests')
    parser.add_argument(
        '-x',
        '--exclude',
        nargs='+',
        choices=TESTS,
        metavar='TESTS',
        default=[],
        help='select a set of tests to exclude')
    return parser.parse_args()


def get_base_command(args):
    """Return the base running command."""
    if args.coverage:
        executable = ['coverage', 'run', '--parallel-mode']
    else:
        executable = [sys.executable]
    return executable


def get_selected_tests(args, tests, sources):
    """Return the selected tests."""
    for exclude_test in args.exclude:
        tests_copy = tests[:]
        for i, test in enumerate(tests_copy):
            if test.startswith(exclude_test):
                tests.pop(i)
                sources.pop(i)
    return tests, sources


def main():
    """The main procedure."""
    args = parse_args()
    base_command = get_base_command(args)
    tests, sources = get_selected_tests(args, TESTS, SOURCES)
    for i, test in enumerate(tests):
        command = base_command[:]
        if args.coverage:
            if sources[i]:
                command.extend(['--source ', sources[i]])
        command.append(test + '.py')
        if args.verbose:
            command.append('--verbose')
        elif args.quiet:
            command.append('--quiet')
        subprocess.call(' '.join(command), shell=True)
    if args.coverage:
        subprocess.call(['coverage', 'combine'])
        subprocess.call(['coverage', 'html'])


if __name__ == '__main__':
    main()
