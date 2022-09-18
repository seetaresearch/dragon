Operator
========

.. doxygenclass:: dragon::Operator

Constructors
------------

.. doxygenfunction:: dragon::Operator::Operator(const OperatorDef &def, Workspace *ws)

Public Properties
-----------------

arg
###
.. doxygenfunction:: dragon::Operator::arg

args
####
.. doxygenfunction:: dragon::Operator::args

data_format
###########
.. doxygenfunction:: dragon::Operator::data_format

data_type
#########
.. doxygenfunction:: dragon::Operator::data_type

def
###
.. doxygenfunction:: dragon::Operator::def

name
####
.. doxygenfunction:: dragon::Operator::name

phase
#####
.. doxygenfunction:: dragon::Operator::phase

type
####
.. doxygenfunction:: dragon::Operator::type

workspace
#########
.. doxygenfunction:: dragon::Operator::workspace

Public Functions
----------------

DeriveFrom
##########
.. doxygenfunction:: dragon::Operator::DeriveFrom

Fuse
####
.. doxygenfunction:: dragon::Operator::Fuse

GetArgument
###########
.. doxygenfunction:: dragon::Operator::GetArgument(const string &name)

GetArgument
###########
.. doxygenfunction:: dragon::Operator::GetArgument(const string &name, const T &default_value)

Input
#####
.. doxygenfunction:: dragon::Operator::Input(int index)

Input
#####
.. doxygenfunction:: dragon::Operator::Input(const string &name)

InputSize
#########
.. doxygenfunction:: dragon::Operator::InputSize

MessageForUnsupported
#####################
.. doxygenfunction:: dragon::Operator::MessageForUnsupported

Output
######
.. doxygenfunction:: dragon::Operator::Output(int index)

Output
######
.. doxygenfunction:: dragon::Operator::Output(int index, const vector<int> &inputs_at)

Output
######
.. doxygenfunction:: dragon::Operator::Output(const string &name)

OutputSize
##########
.. doxygenfunction:: dragon::Operator::OutputSize

Run
###
.. doxygenfunction:: dragon::Operator::Run

.. raw:: html

  <style>
    h1:before {
      content: "dragon::";
      color: #103d3e;
    }
  </style>
