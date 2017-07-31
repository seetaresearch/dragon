CIFAR-10 [Caffe1 style]
=====================================

### Runtime Requirements for Python

0. Package: lmdb
1. Package: python-opencv

-----

Prepare the Dataset
-------------------

- download ``cifar-10-python.tar.gz`` from [http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

- copy to data folder

```Shell
cp cifar-10-python.tar.gz cifar/data
```

- gen db files

```Shell
cd cifar10
python gen_lmdb.py
```

Train "Quick/Full" Model
-------------------

- Quick

```Shell
cd cifar10
python solve_quick.py
```

- Full

```Shell
cd cifar10
python solve_full.py
```

Infer "Quick" Model after Training
-------------------

```Shell
cd cifar10
python infer.py
```
