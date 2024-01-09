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
"""Python setup script."""

import argparse
import os
import shutil
import subprocess
import sys

import setuptools
import setuptools.command.build_py
import setuptools.command.install
import wheel.bdist_wheel


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None)
    args, unknown = parser.parse_known_args()
    args.git_version = None
    args.long_description = ""
    sys.argv = [sys.argv[0]] + unknown
    if args.version is None and os.path.exists("version.txt"):
        with open("version.txt", "r") as f:
            args.version = f.read().strip()
    if os.path.exists(".git"):
        try:
            git_version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="./")
            args.git_version = git_version.decode("ascii").strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if os.path.exists("README.md"):
        with open(os.path.join("README.md"), encoding="utf-8") as f:
            args.long_description = f.read()
    return args


def clean_builds():
    """Clean the builds."""
    if os.path.exists("build/lib"):
        shutil.rmtree("build/lib")
    if os.path.exists("seeta_dragon.egg-info"):
        shutil.rmtree("seeta_dragon.egg-info")


def find_libraries(build_lib):
    """Return the pre-built libraries."""
    in_prefix = "" if sys.platform == "win32" else "lib"
    in_suffix = out_suffix = ".so"
    if sys.platform == "win32":
        in_suffix, out_suffix = ".dll", ".pyd"
    elif sys.platform == "darwin":
        in_suffix = ".dylib"
    libraries = {
        "targets/native/lib/{}dragon{}".format(in_prefix, in_suffix): build_lib
        + "/dragon/lib/{}dragon{}".format(in_prefix, in_suffix),
        "targets/native/lib/{}dragon_python{}".format(in_prefix, in_suffix): build_lib
        + "/dragon/lib/libdragon_python{}".format(out_suffix),
    }
    if sys.platform == "win32":
        libraries["targets/native/lib/dragon.lib"] = build_lib + "/dragon/lib/dragon.lib"
        libraries["targets/native/lib/protobuf.lib"] = build_lib + "/dragon/lib/protobuf.lib"
    return libraries


class BuildPyCommand(setuptools.command.build_py.build_py):
    """Enhanced 'build_py' command."""

    def build_packages(self):
        shutil.copytree("dragon/python", self.build_lib + "/dragon")
        shutil.copytree("dali", self.build_lib + "/dragon/vm/dali")
        shutil.copytree("keras", self.build_lib + "/dragon/vm/keras")
        shutil.copytree("tensorflow", self.build_lib + "/dragon/vm/tensorflow")
        shutil.copytree("tensorrt/python", self.build_lib + "/dragon/vm/tensorrt")
        shutil.copytree("torch", self.build_lib + "/dragon/vm/torch")
        shutil.copytree("torchvision", self.build_lib + "/dragon/vm/torchvision")
        with open(self.build_lib + "/dragon/version.py", "w") as f:
            f.write(
                'version = "{}"\n'
                'git_version = "{}"\n'
                "__version__ = version\n".format(args.version, args.git_version)
            )

    def build_package_data(self):
        shutil.copytree("targets/native/include", self.build_lib + "/dragon/include")
        if not os.path.exists(self.build_lib + "/dragon/lib"):
            os.makedirs(self.build_lib + "/dragon/lib")
        for src, dest in find_libraries(self.build_lib).items():
            if os.path.exists(src):
                shutil.copy(src, dest)
            else:
                print("ERROR: Unable to find the library at <%s>." % src)
                sys.exit()


class InstallCommand(setuptools.command.install.install):
    """Enhanced 'install' command."""

    def initialize_options(self):
        super(InstallCommand, self).initialize_options()
        self.old_and_unmanageable = True


class WheelDistCommand(wheel.bdist_wheel.bdist_wheel):
    """Enhanced 'bdist_wheel' command."""

    def finalize_options(self):
        super(WheelDistCommand, self).finalize_options()
        self.root_is_pure = False


args = parse_args()
setuptools.setup(
    name="seeta-dragon",
    version=args.version,
    description="Dragon: A Computation Graph Virtual Machine " "Based Deep Learning Framework",
    long_description=args.long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seetaresearch/dragon",
    author="SeetaTech",
    license="BSD 2-Clause",
    packages=["dragon"],
    cmdclass={
        "build_py": BuildPyCommand,
        "install": InstallCommand,
        "bdist_wheel": WheelDistCommand,
    },
    python_requires=">=3.8",
    install_requires=["numpy>=1.22.0", "protobuf>=3.20.0", "ml_dtypes>=0.2.0"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
clean_builds()
