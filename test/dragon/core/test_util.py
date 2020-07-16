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
"""Test the util module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import logging
import os
import threading
import unittest

import dragon

from dragon.core.proto import dragon_pb2
from dragon.core.util import deprecation
from dragon.core.util import nest
from dragon.core.util import registry
from dragon.core.util import serialization
from dragon.core.util import tls
from dragon.core.testing.unittest.common_utils import run_tests


class TestDeprecation(unittest.TestCase):
    """Test the deprecation utility."""

    try:
        @deprecation.deprecated('2077-01-01', 'Bad deprecated.')
        @property
        def deprecated_property(self):
            return None
    except ValueError:
        pass

    def test_deprecated(self):
        @deprecation.deprecated(
            '2077-01-01', 'This function is deprecated since 2077.')
        def func():
            pass
        dragon.logging.set_verbosity('FATAL')
        func()
        dragon.logging.set_verbosity('INFO')
        try:
            @deprecation.deprecated('2077-01', 'Bad deprecated.')
            def func():
                pass
            func()
        except ValueError:
            pass
        try:
            @deprecation.deprecated('2077-01-01', '')
            def func():
                pass
            func()
        except ValueError:
            pass

    def test_not_installed(self):
        try:
            deprecation.not_installed('!@#$%^&**()')()
        except ImportError:
            pass
        module = deprecation.NotInstalled('!@#$%^&**()')
        try:
            module.func()
        except ImportError:
            pass


class TestLogging(unittest.TestCase):
    """Test the logging utility."""

    def test_message(self):
        dragon.logging.set_verbosity('FATAL')
        self.assertEqual(dragon.logging.get_verbosity(), logging.FATAL)
        dragon.logging.debug('Test dragon.logging.debug(..)')
        dragon.logging.info('Test dragon.logging.info(..)')
        dragon.logging.warning('Test dragon.logging.warning(..)')
        dragon.logging.error('Test dragon.logging.error(..)')
        dragon.logging.log('INFO', 'Test dragon.logging.error(..)')
        dragon.logging.set_verbosity('INFO')
        self.assertEqual(dragon.logging.get_verbosity(), logging.INFO)

    def test_logging_file(self):
        dragon.logging.set_directory(None)


class TestNest(unittest.TestCase):
    """Test the nest utility."""

    def test_nested(self):
        self.assertTrue(nest.is_nested(list()))
        self.assertTrue(nest.is_sequence(list()))
        self.assertTrue(nest.is_nested(tuple()))
        self.assertTrue(nest.is_sequence(tuple()))
        self.assertTrue(nest.is_nested(dict()))
        self.assertFalse(nest.is_sequence(dict()))

    def test_flatten(self):
        self.assertEqual(nest.flatten(1), [1])
        for a, b in zip(nest.flatten_with_paths([2, 4, [31, 32], 1]),
                        [((0,), 2), ((1,), 4), ((2, 0), 31), ((2, 1), 32), ((3,), 1)]):
            self.assertEqual(a, b)
        for a, b in zip(nest.flatten_with_paths({2: 2, 4: 4, 3: {1: 31, 2: 32}, 1: 1}),
                        [((1,), 1), ((2,), 2), ((3, 1), 31), ((3, 2), 32), ((4,), 4)]):
            self.assertEqual(a, b)


class TestRegistry(unittest.TestCase):
    """Test the registry utility."""

    def test_register(self):
        reg = registry.Registry('test_registry')
        reg.register('a+b', lambda a, b: a + b)
        self.assertTrue('a+b' in reg.keys)
        self.assertTrue(reg.has('a+b'))
        self.assertEqual(reg.get('a+b')(1, 2), 3)
        try:
            reg.get('c+d')
        except KeyError:
            pass


class TestSerialization(unittest.TestCase):
    """Test the serialization utility."""

    def test_bytes(self):
        f = io.BytesIO(b'123')
        serialization.save_bytes(b'456', f)
        f.seek(0)
        self.assertEqual(serialization.load_bytes(f), b'456')
        save_file = '/tmp/test_dragon_serialization_save_bytes'
        try:
            serialization.save_bytes(b'789', save_file)
        except OSError:
            pass
        try:
            s = serialization.load_bytes(save_file)
            self.assertEqual(s, b'789')
        except FileNotFoundError:
            pass
        try:
            if os.path.exists(save_file):
                os.remove(save_file)
        except PermissionError:
            pass

    def test_proto(self):
        self.assertEqual(serialization.serialize_proto(None), b'')
        s = serialization.serialize_proto(dragon_pb2.OperatorDef(name='!@#$%^&**()'))
        s = serialization.serialize_proto(s)
        proto = serialization.deserialize_proto(s, dragon_pb2.OperatorDef())
        self.assertEqual(proto, dragon_pb2.OperatorDef(name='!@#$%^&**()'))
        try:
            serialization.serialize_proto(1)
        except ValueError:
            pass
        try:
            serialization.deserialize_proto(2, dragon_pb2.OperatorDef())
        except ValueError:
            pass
        try:
            serialization.deserialize_proto(s, 2)
        except ValueError:
            pass


class TestTLS(unittest.TestCase):
    """Test the tls utility."""

    def test_constant(self):
        def write(i, q):
            c.value = i
            q.append(c.value)
        c, q = tls.Constant(value=-1), []
        threads = [threading.Thread(target=write, args=[i, q]) for i in range(4)]
        for t in threads:
            t.start()
            t.join()
        self.assertEqual(c.value, -1)

    def test_stack(self):
        s = tls.Stack()
        s.enforce_nesting = True
        self.assertEqual(s.enforce_nesting, True)
        try:
            with s.get_controller('!@#$%^&**()'):
                s.push('123456')
        except RuntimeError:
            pass
        s.enforce_nesting = False
        with s.get_controller('!@#$%^&**()'):
            s.push('123456')


if __name__ == '__main__':
    run_tests()
