# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import subprocess

import argparse

TESTS_AND_SOURCES = [
    ('dragon/core/test_ops', 'dragon.core.ops'),
]

TESTS = [t[0] for t in TESTS_AND_SOURCES]
SOURCES = [t[1] for t in TESTS_AND_SOURCES]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the unittests',
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
    return parser.parse_args()


def get_base_command(args):
    """Return the base running command."""
    if args.coverage:
        executable = ['coverage', 'run', '--parallel-mode']
    else:
        executable = [sys.executable]
    return executable


def main():
    """The main procedure."""
    args = parse_args()
    base_command = get_base_command(args)
    for i, test in enumerate(TESTS):
        command = base_command[:]
        if args.coverage:
            if SOURCES[i]:
                command.extend(['--source ', SOURCES[i]])
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
