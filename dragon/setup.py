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
import setuptools
import setuptools.command.install
import shutil
import subprocess
import sys

try:
    # Override a non-pure "wheel" for pybind distributions
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None


# Read the current version info
with open('version.txt', 'r') as f:
    version = f.read().strip()
try:
    git_version = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd='../').decode('ascii').strip()
except (OSError, subprocess.CalledProcessError):
    git_version = None


def clean():
    """Remove the work directories."""
    if os.path.exists('dragon/version.py'):
        shutil.rmtree('dragon')
    if os.path.exists('seeta_dragon.egg-info'):
        shutil.rmtree('seeta_dragon.egg-info')


def configure():
    """Prepare the package files."""
    clean()
    # Create a temporary site-package directory.
    shutil.copytree('python', 'dragon')
    # Copy headers.
    shutil.copytree('../targets/native/include', 'dragon/include')
    # Copy "caffe" => "dragon.vm.caffe"
    shutil.copytree('../caffe', 'dragon/vm/caffe')
    # Copy "dali" => "dragon.vm.dali"
    shutil.copytree('../dali', 'dragon/vm/dali')
    # Copy "tensorflow" => "dragon.vm.tensorflow"
    shutil.copytree('../tensorflow', 'dragon/vm/tensorflow')
    # Copy "tensorlayer" => "dragon.vm.tensorlayer"
    shutil.copytree('../tensorlayer', 'dragon/vm/tensorlayer')
    # Copy "tensorrt/python" => "dragon.vm.tensorrt"
    shutil.copytree('../tensorrt/python', 'dragon/vm/tensorrt')
    # Copy "torch" => "dragon.vm.torch"
    shutil.copytree('../torch', 'dragon/vm/torch')
    # Copy "torchvision" => "dragon.vm.torchvision"
    shutil.copytree('../torchvision', 'dragon/vm/torchvision')
    # Copy the pre-built libraries.
    os.makedirs('dragon/lib')
    for src, dest in find_libraries().items():
        if os.path.exists(src):
            shutil.copy(src, dest)
        else:
            print('ERROR: Unable to find the library at <%s>.\n'
                  'Build it before installing to package.' % src)
            shutil.rmtree('dragon')
            sys.exit()
    # Write the version file.
    with open('dragon/version.py', 'w') as f:
        f.write("from __future__ import absolute_import\n"
                "from __future__ import division\n"
                "from __future__ import print_function\n\n"
                "version = '{}'\n"
                "git_version = '{}'\n".format(version, git_version))


class install(setuptools.command.install.install):
    """Old-style command to prevent from installing egg."""

    def run(self):
        setuptools.command.install.install.run(self)


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
        'dragon/lib/libdragon_python{}'.format(out_suffix)
    }
    if sys.platform == 'win32':
        libraries['../targets/native/lib/dragon.lib'] = 'dragon/lib/dragon.lib'
        libraries['../targets/native/lib/protobuf.lib'] = 'dragon/lib/protobuf.lib'
    return libraries


def find_packages():
    """Return the python sources installed to package."""
    packages = []
    for root, _, files in os.walk('dragon'):
        if os.path.exists(os.path.join(root, '__init__.py')):
            packages.append(root)
    return packages


def find_package_data():
    """Return the external data installed to package."""
    headers, libraries = [], []
    for root, _, files in os.walk('dragon/include'):
        root = root[len('dragon/'):]
        for file in files:
            headers.append(os.path.join(root, file))
    for root, _, files in os.walk('dragon/lib'):
        root = root[len('dragon/'):]
        for file in files:
            libraries.append(os.path.join(root, file))
    return headers + libraries


configure()
setuptools.setup(
    name='seeta-dragon',
    version=version,
    description='Dragon: A Computation Graph Virtual Machine '
                'Based Deep Learning Framework',
    url='https://github.com/seetaresearch/dragon',
    author='SeetaTech',
    license='BSD 2-Clause',
    packages=find_packages(),
    package_data={'dragon': find_package_data()},
    package_dir={'dragon': 'dragon'},
    cmdclass={'bdist_wheel': bdist_wheel, 'install': install},
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    install_requires=['numpy', 'protobuf', 'kpl-dataset'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
clean()
