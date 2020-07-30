Operator
========

.. doxygenclass:: dragon::Operator

Constructors
------------

.. doxygenfunction:: dragon::Operator::Operator(const OperatorDef &def, Workspace *ws)

Public Functions
----------------

Arg
###
.. doxygenfunction:: dragon::Operator::Arg

Args
####
.. doxygenfunction:: dragon::Operator::Args

Buffer
######
.. doxygenfunction:: dragon::Operator::Buffer

Fuse
####
.. doxygenfunction:: dragon::Operator::Fuse

Input
#####
.. doxygenfunction:: dragon::Operator::Input

InputSize
#########
.. doxygenfunction:: dragon::Operator::InputSize

Output
######
.. doxygenfunction:: dragon::Operator::Output(int i)

MessageForUnsupported
#####################
.. doxygenfunction:: dragon::Operator::MessageForUnsupported

Output
######
.. doxygenfunction:: dragon::Operator::Output(int i, const vec32_t &inputs)

OutputSize
##########
.. doxygenfunction:: dragon::Operator::OutputSize

Run
###
.. doxygenfunction:: dragon::Operator::Run

UpdateFrom
##########
.. doxygenfunction:: dragon::Operator::UpdateFrom

data_format
###########
.. doxygenfunction:: dragon::Operator::data_format

arg
###
.. doxygenfunction:: dragon::Operator::arg

args
####
.. doxygenfunction:: dragon::Operator::args

def
###
.. doxygenfunction:: dragon::Operator::def

dtype
#####
.. doxygenfunction:: dragon::Operator::dtype

handle
######
.. doxygenfunction:: dragon::Operator::handle

name
####
.. doxygenfunction:: dragon::Operator::name

type
####
.. doxygenfunction:: dragon::Operator::type

phase
#####
.. doxygenfunction:: dragon::Operator::phase

ws
##
.. doxygenfunction:: dragon::Operator::ws

.. raw:: html

  <style>
    h1:before {
      content: "dragon::";
      color: #103d3e;
    }
  </style>
