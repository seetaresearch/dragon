Tensor
======

.. autoclass:: dragon.vm.torch.Tensor

Properties
----------

data
####
.. autoattribute:: dragon.vm.torch.Tensor.data

dtype
#####
.. autoattribute:: dragon.vm.torch.Tensor.dtype

device
######
.. autoattribute:: dragon.vm.torch.Tensor.device

grad
####
.. autoattribute:: dragon.vm.torch.Tensor.grad

id
###
.. autoattribute:: dragon.vm.torch.Tensor.id

is_leaf
#######
.. autoattribute:: dragon.vm.torch.Tensor.is_leaf

requires_grad
#############
.. autoattribute:: dragon.vm.torch.Tensor.requires_grad

shape
#####
.. autoattribute:: dragon.vm.torch.Tensor.shape

Methods
-------

abs
###
.. automethod:: dragon.vm.torch.Tensor.abs

add
###
.. automethod:: dragon.vm.torch.Tensor.add

add\_
#####
.. automethod:: dragon.vm.torch.Tensor.add_

argmax
######
.. automethod:: dragon.vm.torch.Tensor.argmax

argmin
######
.. automethod:: dragon.vm.torch.Tensor.argmin

backward
########
.. automethod:: dragon.vm.torch.Tensor.backward

bitwise_not
###########
.. automethod:: dragon.vm.torch.Tensor.bitwise_not

bitwise_not\_
#############
.. automethod:: dragon.vm.torch.Tensor.bitwise_not_

bitwise_xor
###########
.. automethod:: dragon.vm.torch.Tensor.bitwise_xor

bitwise_xor\_
#############
.. automethod:: dragon.vm.torch.Tensor.bitwise_xor_

bool
####
.. automethod:: dragon.vm.torch.Tensor.bool

bool\_
######
.. automethod:: dragon.vm.torch.Tensor.bool_

byte
####
.. automethod:: dragon.vm.torch.Tensor.byte

byte\_
######
.. automethod:: dragon.vm.torch.Tensor.byte_

ceil
####
.. automethod:: dragon.vm.torch.Tensor.ceil

ceil\_
######
.. automethod:: dragon.vm.torch.Tensor.ceil_

char
####
.. automethod:: dragon.vm.torch.Tensor.char

char\_
######
.. automethod:: dragon.vm.torch.Tensor.char_

chunk
#####
.. automethod:: dragon.vm.torch.Tensor.chunk

clamp
#####
.. automethod:: dragon.vm.torch.Tensor.clamp

clamp\_
#######
.. automethod:: dragon.vm.torch.Tensor.clamp_

copy\_
######
.. automethod:: dragon.vm.torch.Tensor.copy_

cos
###
.. automethod:: dragon.vm.torch.Tensor.cos

cpu
###
.. automethod:: dragon.vm.torch.Tensor.cpu

cuda
####
.. automethod:: dragon.vm.torch.Tensor.cuda

cumsum
######
.. automethod:: dragon.vm.torch.Tensor.cumsum

detach
######
.. automethod:: dragon.vm.torch.Tensor.detach

dim
###
.. automethod:: dragon.vm.torch.Tensor.dim

div
###
.. automethod:: dragon.vm.torch.Tensor.div

div\_
######
.. automethod:: dragon.vm.torch.Tensor.div_

double
######
.. automethod:: dragon.vm.torch.Tensor.double

double\_
########
.. automethod:: dragon.vm.torch.Tensor.double_

eq
###
.. automethod:: dragon.vm.torch.Tensor.eq

exp
###
.. automethod:: dragon.vm.torch.Tensor.exp

expand
######
.. automethod:: dragon.vm.torch.Tensor.expand

expand_as
#########
.. automethod:: dragon.vm.torch.Tensor.expand_as

fill\_
#######
.. automethod:: dragon.vm.torch.Tensor.fill_

float
#####
.. automethod:: dragon.vm.torch.Tensor.float

float\_
#######
.. automethod:: dragon.vm.torch.Tensor.float_

floor
#####
.. automethod:: dragon.vm.torch.Tensor.floor

floor\_
#######
.. automethod:: dragon.vm.torch.Tensor.floor_

ge
###
.. automethod:: dragon.vm.torch.Tensor.ge

gt
###
.. automethod:: dragon.vm.torch.Tensor.gt

half
####
.. automethod:: dragon.vm.torch.Tensor.half

half\_
######
.. automethod:: dragon.vm.torch.Tensor.half_

index_select
############
.. automethod:: dragon.vm.torch.Tensor.index_select

int
###
.. automethod:: dragon.vm.torch.Tensor.int

int\_
######
.. automethod:: dragon.vm.torch.Tensor.int_

is_floating_point
#################
.. automethod:: dragon.vm.torch.Tensor.is_floating_point

le
###
.. automethod:: dragon.vm.torch.Tensor.le

log
###
.. automethod:: dragon.vm.torch.Tensor.log

logsumexp
#########
.. automethod:: dragon.vm.torch.Tensor.logsumexp

long
####
.. automethod:: dragon.vm.torch.Tensor.long

long\_
######
.. automethod:: dragon.vm.torch.Tensor.long_

lt
###
.. automethod:: dragon.vm.torch.Tensor.lt

masked_fill\_
#############
.. automethod:: dragon.vm.torch.Tensor.masked_fill_

max
###
.. automethod:: dragon.vm.torch.Tensor.max

masked_select
#############
.. automethod:: dragon.vm.torch.Tensor.masked_select

mean
####
.. automethod:: dragon.vm.torch.Tensor.mean

min
###
.. automethod:: dragon.vm.torch.Tensor.min

mul
###
.. automethod:: dragon.vm.torch.Tensor.mul

mul\_
#####
.. automethod:: dragon.vm.torch.Tensor.mul_

multinomial
###########
.. automethod:: dragon.vm.torch.Tensor.multinomial

narrow
######
.. automethod:: dragon.vm.torch.Tensor.narrow

ndimension
##########
.. automethod:: dragon.vm.torch.Tensor.ndimension

ne
###
.. automethod:: dragon.vm.torch.Tensor.ne

nonzero
#######
.. automethod:: dragon.vm.torch.Tensor.nonzero

normal\_
########
.. automethod:: dragon.vm.torch.Tensor.normal_

numel
#####
.. automethod:: dragon.vm.torch.Tensor.numel

numpy
#####
.. automethod:: dragon.vm.torch.Tensor.numpy

one\_
#####
.. automethod:: dragon.vm.torch.Tensor.one_

permute
#######
.. automethod:: dragon.vm.torch.Tensor.permute

pow
###
.. automethod:: dragon.vm.torch.Tensor.pow

reciprocal
##########
.. automethod:: dragon.vm.torch.Tensor.reciprocal

reciprocal\_
############
.. automethod:: dragon.vm.torch.Tensor.reciprocal_

repeat
######
.. automethod:: dragon.vm.torch.Tensor.repeat

reshape
#######
.. automethod:: dragon.vm.torch.Tensor.reshape

reshape\_
#########
.. automethod:: dragon.vm.torch.Tensor.reshape_

retain_grad
###########
.. automethod:: dragon.vm.torch.Tensor.retain_grad

round
#####
.. automethod:: dragon.vm.torch.Tensor.round

round\_
#######
.. automethod:: dragon.vm.torch.Tensor.round_

rsqrt
#####
.. automethod:: dragon.vm.torch.Tensor.rsqrt

rsqrt\_
#######
.. automethod:: dragon.vm.torch.Tensor.rsqrt_

sign
####
.. automethod:: dragon.vm.torch.Tensor.sign

sign\_
######
.. automethod:: dragon.vm.torch.Tensor.sign_

sin
###
.. automethod:: dragon.vm.torch.Tensor.sin

size
####
.. automethod:: dragon.vm.torch.Tensor.size

sqrt
####
.. automethod:: dragon.vm.torch.Tensor.sqrt

sqrt\_
######
.. automethod:: dragon.vm.torch.Tensor.sqrt_

squeeze
#######
.. automethod:: dragon.vm.torch.Tensor.squeeze

squeeze\_
#########
.. automethod:: dragon.vm.torch.Tensor.squeeze_

sum
###
.. automethod:: dragon.vm.torch.Tensor.sum

sub
###
.. automethod:: dragon.vm.torch.Tensor.sub

sub\_
#####
.. automethod:: dragon.vm.torch.Tensor.sub_

topk
####
.. automethod:: dragon.vm.torch.Tensor.topk

type
####
.. automethod:: dragon.vm.torch.Tensor.type

uniform\_
#########
.. automethod:: dragon.vm.torch.Tensor.uniform_

unsqueeze
#########
.. automethod:: dragon.vm.torch.Tensor.unsqueeze

unsqueeze\_
###########
.. automethod:: dragon.vm.torch.Tensor.unsqueeze_

view
####
.. automethod:: dragon.vm.torch.Tensor.view

view\_
######
.. automethod:: dragon.vm.torch.Tensor.view_

view_as
#######
.. automethod:: dragon.vm.torch.Tensor.view_as

where
#####
.. automethod:: dragon.vm.torch.Tensor.where

zero\_
######
.. automethod:: dragon.vm.torch.Tensor.zero_

.. _torch.abs(...): abs.html
.. _torch.add(...): add.html
.. _torch.argmax(...): argmax.html
.. _torch.argmin(...): argmin.html
.. _torch.bitwise_not(...): bitwise_not.html
.. _torch.bitwise_xor(...): bitwise_xor.html
.. _torch.ceil(...): ceil.html
.. _torch.clamp(...): clamp.html
.. _torch.cos(...): cos.html
.. _torch.cumsum(...): cumsum.html
.. _torch.div(...): div.html
.. _torch.eq(...): eq.html
.. _torch.exp(...): exp.html
.. _torch.expand(...): expand.html
.. _torch.floor(...): floor.html
.. _torch.ge(...): ge.html
.. _torch.gt(...): gt.html
.. _torch.le(...): le.html
.. _torch.lt(...): lt.html
.. _torch.mul(...): mul.html
.. _torch.ne(...): ne.html
.. _torch.neg(...): neg.html
.. _torch.pow(...): pow.html
.. _torch.reciprocal(...): reciprocal.html
.. _torch.reshape(...): reshape.html
.. _torch.round(...): round.html
.. _torch.rsqrt(...): rsqrt.html
.. _torch.sign(...): sign.html
.. _torch.sin(...): sin.html
.. _torch.sqrt(...): sqrt.html
.. _torch.sub(...): sub.html
.. _torch.topk(...): topk.html

.. raw:: html

  <style>
    h1:before {
      content: "torch.";
      color: #103d3e;
    }
  </style>
