from distutils.core import setup
import os.path, sys
import shutil

packages = []

def find_packages(root_dir):
    filenames = os.listdir(root_dir)
    for filename in filenames:
        filepath = os.path.join(root_dir, filename)
        if os.path.isdir(filepath):
            find_packages(filepath)
        else:
            if filename == '__init__.py':
                packages.append(root_dir)

def find_modules():
    dragon_c_lib_win32 = '../lib/dragon.dll'
    dragon_c_lib_other = '../lib/libdragon.so'
    if os.path.exists(dragon_c_lib_win32):
        shutil.copy(dragon_c_lib_win32, 'dragon/libdragon.pyd')
    elif os.path.exists(dragon_c_lib_other):
        shutil.copy(dragon_c_lib_other, 'dragon/libdragon.so')
    else:
        print('ERROR: Unable to find modules. built Dragon using CMake.')
        sys.exit()


def find_resources():
    c_lib = ['libdragon.*']
    protos = ['protos/*.proto', 'vm/caffe/proto/*.proto']
    others = []
    return c_lib + protos + others

find_packages('dragon')
find_modules()

setup(name = 'dragon',
      version='0.2.2.2',
      description = 'Dragon: A Computation Graph Virtual Machine Based Deep Learning Framework',
      url='https://github.com/seetaresearch/Dragon',
      author='Ting Pan',
      license='BSD 2-Clause',
      packages=packages,
      package_dir={'dragon': 'dragon'},
      package_data={'dragon': find_resources()})
