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
#     <https://github.com/pytorch/pytorch/blob/master/torch/utils/cpp_extension.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import copy as _copy
import glob as _glob
import os as _os
import re as _re
import subprocess as _subprocess
import sys as _sys

from setuptools import Extension as _Extension
from setuptools.command.build_ext import build_ext as _build_ext

from dragon.core.device import cuda as _cuda


def include_paths(cuda=False):
    """Return the path of headers.

    Parameters
    ----------
    cuda : bool, optional, default=False
        ``True`` to add cuda paths.

    Returns
    -------
    Sequence[str]
        The path sequence.

    """
    here = _os.path.abspath(__file__)
    package_path = _os.path.dirname(_os.path.dirname(here))
    paths = [_os.path.join(package_path, 'include')]
    if cuda:
        cuda_home_include = _join_cuda_path('include')
        if cuda_home_include != '/usr/include':
            paths.append(cuda_home_include)
            if not _os.path.exists(cuda_home_include + '/cub'):
                paths.append(paths[0] + '/cub')
        if CUDNN_HOME is not None:
            paths.append(_os.path.join(CUDNN_HOME, 'include'))
    return paths


def library_paths(cuda=False):
    """Return the path of libraries.

    Parameters
    ----------
    cuda : bool, optional, default=False
        ``True`` to add cuda paths.

    Returns
    -------
    Sequence[str]
        The path sequence.

    """
    here = _os.path.abspath(__file__)
    package_path = _os.path.dirname(_os.path.dirname(here))
    paths = [_os.path.join(package_path, 'lib')]
    if cuda:
        if IS_WINDOWS:
            lib_dir = 'lib/x64'
        else:
            lib_dir = 'lib64'
            if (not _os.path.exists(_join_cuda_path(lib_dir)) and
                    _os.path.exists(_join_cuda_path('lib'))):
                lib_dir = 'lib'
        paths.append(_join_cuda_path(lib_dir))
        if CUDNN_HOME is not None:
            paths.append(_os.path.join(CUDNN_HOME, lib_dir))
    return paths


class BuildExtension(_build_ext):
    """Custom command to build extensions."""

    user_options = _build_ext.user_options
    boolean_options = _build_ext.boolean_options

    user_options.extend([
        ('no-python-abi-suffix=', None,
         "remove the python abi suffix"),
    ])

    boolean_options.extend(['no-python-abi-suffix'])

    def __init__(self, *args, **kwargs):
        super(BuildExtension, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = 1

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        self.compiler.src_extensions += ['.cu', '.cuh', '.mm']
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile
        original_object_filenames = self.compiler.object_filenames

        def object_filenames(source_filenames, strip_dir, output_dir):
            """Patch to make the objects unique."""
            objects = original_object_filenames(source_filenames, strip_dir, output_dir)
            for i, src_name in enumerate(source_filenames):
                if _os.path.splitext(src_name)[1] in ['.cu', '.cuh', '.mm']:
                    _, src_ext = _os.path.splitext(src_name)
                    obj_base, obj_ext = _os.path.splitext(objects[i])
                    objects[i] = obj_base + src_ext + obj_ext
            return objects

        def unix_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            """Patch to support custom sources."""
            original_compiler = self.compiler.compiler_so
            try:
                cflags = _copy.deepcopy(extra_postargs)
                if _os.path.splitext(src)[1] in ['.cu', '.cuh']:
                    nvcc = [_join_cuda_path('bin', 'nvcc')]
                    self.compiler.set_executable('compiler_so', nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    cflags = (COMMON_NVCC_FLAGS +
                              ['--compiler-options', "'-fPIC'"] +
                              cflags + _get_cuda_arch_flags(cflags))
                else:
                    if isinstance(cflags, dict):
                        cflags = cflags['cxx']
                    cflags += COMMON_CC_FLAGS
                if not any(flag.startswith('-std=') for flag in cflags):
                    cflags.append('-std=c++14')
                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                self.compiler.set_executable('compiler_so', original_compiler)

        def win_compile(
            sources,
            output_dir=None,
            macros=None,
            include_dirs=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
        ):
            compile_info = self.compiler._setup_compile(
                output_dir, macros, include_dirs, sources, depends, extra_postargs)
            _, _, _, pp_opts, _ = compile_info
            self.cflags = _copy.deepcopy(extra_postargs)
            extra_postargs = None

            def spawn(cmd):
                # Using regex to match src, obj and include files.
                src_regex = _re.compile('/T(p|c)(.*)')
                src_list = [m.group(2) for m in (src_regex.match(elem) for elem in cmd) if m]
                obj_regex = _re.compile('/Fo(.*)')
                obj_list = [m.group(1) for m in (obj_regex.match(elem) for elem in cmd) if m]
                include_regex = _re.compile(r'((\-|\/)I.*)')
                include_list = [m.group(1) for m in (include_regex.match(elem) for elem in cmd) if m]
                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src, obj = src_list[0], obj_list[0]
                    if _os.path.splitext(src)[1] in ['.cu', '.cuh']:
                        nvcc = _join_cuda_path('bin', 'nvcc')
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags['nvcc']
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []
                        cflags = COMMON_NVCC_FLAGS + cflags + _get_cuda_arch_flags(cflags)
                        for flag in COMMON_MSVC_FLAGS:
                            cflags = ['-Xcompiler', flag] + cflags
                        cmd = [nvcc, '-c', src, '-o', obj] + pp_opts + include_list + cflags
                    elif isinstance(self.cflags, dict):
                        cflags = COMMON_MSVC_FLAGS + self.cflags['cxx']
                        cmd += cflags
                    elif isinstance(self.cflags, list):
                        cflags = COMMON_MSVC_FLAGS + self.cflags
                        cmd += cflags
                if '/MD' in cmd:
                    cmd.remove('/MD')
                return original_spawn(cmd)

            try:
                self.compiler.spawn = spawn
                return original_compile(sources, output_dir, macros,
                                        include_dirs, debug, extra_preargs,
                                        extra_postargs, depends)
            finally:
                self.compiler.spawn = original_spawn

        if self.compiler.compiler_type == 'msvc':
            self.compiler.compile = win_compile
        else:
            self.compiler._compile = unix_compile
        self.compiler.object_filenames = object_filenames
        _build_ext.build_extensions(self)
        self.compiler.object_filenames = original_object_filenames

    def get_ext_filename(self, ext_name):
        ext_filename = super(BuildExtension, self).get_ext_filename(ext_name)
        if self.no_python_abi_suffix > 0 and _sys.version_info >= (3, 0):
            ext_filename_parts = ext_filename.split('.')
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = '.'.join(without_abi)
        return ext_filename

    def get_export_symbols(self, ext):
        return ext.export_symbols


class CppExtension(object):
    """Extension module for generic c++ sources."""

    def __new__(cls, name, sources, *args, **kwargs):
        include_dirs = kwargs.get('include_dirs', [])
        include_dirs += include_paths()
        kwargs['include_dirs'] = include_dirs
        library_dirs = kwargs.get('library_dirs', [])
        library_dirs += library_paths()
        kwargs['library_dirs'] = library_dirs
        libraries = kwargs.get('libraries', [])
        libraries.extend(COMMON_LINK_LIBRARIES + ['dragon'])
        kwargs['libraries'] = libraries
        define_macros = kwargs.get('define_macros', [])
        define_macros.append(('DRAGON_API=' + DLLIMPORT_STR, None))
        kwargs['define_macros'] = define_macros
        kwargs['language'] = 'c++'
        return _Extension(name, sources, *args, **kwargs)


class CUDAExtension(object):
    """Extension module for generic cuda/c++ sources."""

    def __new__(cls, name, sources, *args, **kwargs):
        include_dirs = kwargs.get('include_dirs', [])
        include_dirs += include_paths(cuda=True)
        kwargs['include_dirs'] = include_dirs
        library_dirs = kwargs.get('library_dirs', [])
        library_dirs += library_paths(cuda=True)
        kwargs['library_dirs'] = library_dirs
        libraries = kwargs.get('libraries', [])
        libraries.extend(COMMON_LINK_LIBRARIES + ['cudart', 'dragon'])
        kwargs['libraries'] = libraries
        define_macros = kwargs.get('define_macros', [])
        define_macros.append(('USE_CUDA', None))
        define_macros.append(('DRAGON_API=' + DLLIMPORT_STR, None))
        kwargs['define_macros'] = define_macros
        kwargs['language'] = 'c++'
        return _Extension(name, sources, *args, **kwargs)


class MPSExtension(object):
    """Extension module for generic mps/objc sources."""

    def __new__(cls, name, sources, *args, **kwargs):
        include_dirs = kwargs.get('include_dirs', [])
        include_dirs += include_paths()
        kwargs['include_dirs'] = include_dirs
        library_dirs = kwargs.get('library_dirs', [])
        library_dirs += library_paths()
        kwargs['library_dirs'] = library_dirs
        libraries = kwargs.get('libraries', [])
        libraries.extend(COMMON_LINK_LIBRARIES + ['dragon'])
        kwargs['libraries'] = libraries
        define_macros = kwargs.get('define_macros', [])
        define_macros.append(('USE_MPS', None))
        define_macros.append(('DRAGON_API=' + DLLIMPORT_STR, None))
        kwargs['define_macros'] = define_macros
        kwargs['language'] = 'c++'
        return _Extension(name, sources, *args, **kwargs)


def _find_cuda():
    """Find the cuda root path."""
    cuda_home = _os.environ.get('CUDA_HOME') or _os.environ.get('CUDA_PATH')
    if cuda_home is None:
        try:
            which = 'where' if IS_WINDOWS else 'which'
            nvcc = _subprocess.check_output(
                [which, 'nvcc']).decode().rstrip('\r\n')
            cuda_home = _os.path.dirname(_os.path.dirname(nvcc))
        except _subprocess.CalledProcessError:
            if IS_WINDOWS:
                cuda_homes = _glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not _os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and not _cuda.is_available():
        print("No CUDA runtime is found, using CUDA_HOME='{}'".format(cuda_home))
    return cuda_home


def _get_cuda_arch_flags(cflags=None):
    """Return the CUDA arch flags."""
    if cflags is not None:
        for flag in cflags:
            if 'arch' in flag:
                return []
    supported_arches = ['3.5', '3.7',
                        '5.0', '5.2', '5.3',
                        '6.0', '6.1', '6.2',
                        '7.0', '7.2', '7.5',
                        '8.0', '8.6']
    valid_arch_strings = supported_arches + [s + "+PTX" for s in supported_arches]
    capability = _cuda.get_device_capability()
    arch_list = ['{}.{}'.format(capability[0], capability[1])]

    flags = []
    for arch in arch_list:
        if arch not in valid_arch_strings:
            raise ValueError("Unknown CUDA arch ({}) or GPU not supported".format(arch))
        else:
            num = arch[0] + arch[2]
            flags.append('-gencode=arch=compute_{},code=sm_{}'.format(num, num))
            if arch.endswith('+PTX'):
                flags.append('-gencode=arch=compute_{},code=compute_{}'.format(num, num))

    return list(set(flags))


def _join_cuda_path(*paths):
    """Join given paths with <CUDA_HOME>."""
    if CUDA_HOME is None:
        raise EnvironmentError(
            '<CUDA_HOME> environment variable is not set. '
            'Please set it to your CUDA install root.')
    return _os.path.join(CUDA_HOME, *paths)


IS_WINDOWS = _sys.platform == 'win32'
CUDA_HOME = _find_cuda()
CUDNN_HOME = _os.environ.get('CUDNN_HOME') or _os.environ.get('CUDNN_PATH')
COMMON_CC_FLAGS = ['-Wno-sign-compare', '-Wno-unused-variable', '-Wno-reorder']
COMMON_MSVC_FLAGS = ['/MT', '/EHsc', '/wd4819', '/wd4244', '/wd4251', '/wd4275', '/wd4800', '/wd4996']
COMMON_NVCC_FLAGS = ['-w'] if IS_WINDOWS else ['-std=c++14']
COMMON_LINK_LIBRARIES = ['protobuf'] if IS_WINDOWS else []
DLLIMPORT_STR = '__declspec(dllimport)' if IS_WINDOWS else ''
