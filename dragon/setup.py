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

import os
import shutil
import subprocess
import sys

import setuptools
import setuptools.command.build_py
import setuptools.command.install

try:
    # Override a non-pure "wheel" for pybind distributions.
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super(bdist_wheel, self).finalize_options()
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None


version = git_version = None
with open('version.txt', 'r') as f:
    version = f.read().strip()
if os.path.exists('.git'):
    try:
        git_version = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd='../')
        git_version = git_version.decode('ascii').strip()
    except (OSError, subprocess.CalledProcessError):
        pass


def clean_builds():
    """Clean the builds."""
    if os.path.exists('dragon/version.py'):
        shutil.rmtree('dragon')
    if os.path.exists('build/lib'):
        shutil.rmtree('build/lib')
    if os.path.exists('seeta_dragon.egg-info'):
        shutil.rmtree('seeta_dragon.egg-info')


def find_libraries():
    """Return the pre-built libraries."""
    in_prefix = '' if sys.platform == 'win32' else 'lib'
    in_suffix = out_suffix = '.so'
    if sys.platform == 'win32':
        in_suffix, out_suffix = '.dll', '.pyd'
    elif sys.platform == 'darwin':
        in_suffix = '.dylib'
    libraries = {
        '../targets/native/lib/{}dragon{}'.format(in_prefix, in_suffix):
        'dragon/lib/{}dragon{}'.format(in_prefix, in_suffix),
        '../targets/native/lib/{}dragon_python{}'.format(in_prefix, in_suffix):
        'dragon/lib/libdragon_python{}'.format(out_suffix),
    }
    if sys.platform == 'win32':
        libraries['../targets/native/lib/dragon.lib'] = 'dragon/lib/dragon.lib'
        libraries['../targets/native/lib/protobuf.lib'] = 'dragon/lib/protobuf.lib'
    return libraries


def find_packages(top):
    """Return the python sources installed to package."""
    packages = []
    for root, _, _ in os.walk(top):
        if os.path.exists(os.path.join(root, '__init__.py')):
            packages.append(root)
    return packages


def find_package_data(top):
    """Return the external data installed to package."""
    headers, libraries = [], []
    for root, _, files in os.walk(top + '/include'):
        root = root[len(top + '/'):]
        for file in files:
            headers.append(os.path.join(root, file))
    for root, _, files in os.walk(top + '/lib'):
        root = root[len(top + '/'):]
        for file in files:
            libraries.append(os.path.join(root, file))
    return headers + libraries


class BuildPyCommand(setuptools.command.build_py.build_py):
    """Enhanced 'build_py' command."""

    def build_packages(self):
        clean_builds()
        shutil.copytree('python', 'dragon')
        shutil.copytree('../caffe', 'dragon/vm/caffe')
        shutil.copytree('../dali', 'dragon/vm/dali')
        shutil.copytree('../tensorflow', 'dragon/vm/tensorflow')
        shutil.copytree('../tensorlayer', 'dragon/vm/tensorlayer')
        shutil.copytree('../tensorrt/python', 'dragon/vm/tensorrt')
        shutil.copytree('../torch', 'dragon/vm/torch')
        shutil.copytree('../torchvision', 'dragon/vm/torchvision')
        with open('dragon/version.py', 'w') as f:
            f.write("from __future__ import absolute_import\n"
                    "from __future__ import division\n"
                    "from __future__ import print_function\n\n"
                    "version = '{}'\n"
                    "git_version = '{}'\n".format(version, git_version))
        self.packages = find_packages('dragon')
        super(BuildPyCommand, self).build_packages()

    def build_package_data(self):
        shutil.copytree('../targets/native/include', 'dragon/include')
        if not os.path.exists('dragon/lib'):
            os.makedirs('dragon/lib')
        for src, dest in find_libraries().items():
            if os.path.exists(src):
                shutil.copy(src, dest)
            else:
                print('ERROR: Unable to find the library at <%s>.\n'
                      'Build it before installing to package.' % src)
                sys.exit()
        self.package_data = {'dragon': find_package_data('dragon')}
        super(BuildPyCommand, self).build_package_data()


class InstallCommand(setuptools.command.install.install):
    """Enhanced 'install' command."""

    def run(self):
        # Old-style install instead of egg.
        super(InstallCommand, self).run()


setuptools.setup(
    name='seeta-dragon',
    version=version,
    description='Dragon: A Computation Graph Virtual Machine '
                'Based Deep Learning Framework',
    url='https://github.com/seetaresearch/dragon',
    author='SeetaTech',
    license='BSD 2-Clause',
    packages=find_packages('python'),
    package_dir={'dragon': 'dragon'},
    cmdclass={'bdist_wheel': bdist_wheel,
              'build_py': BuildPyCommand,
              'install': InstallCommand},
    python_requires='>=3.6',
    install_requires=['numpy', 'protobuf', 'kpl-dataset'],
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Programming Language :: C++',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3 :: Only',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Mathematics',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Software Development',
                 'Topic :: Software Development :: Libraries',
                 'Topic :: Software Development :: Libraries :: Python Modules'],
)
clean_builds()
