Workspace
=========

.. doxygenclass:: dragon::Workspace

Constructors
------------

.. doxygenfunction:: dragon::Workspace::Workspace(const string &name)

Public Properties
-----------------

data
####
.. doxygenfunction:: dragon::Workspace::data(size_t size, const string &name = "BufferShared")

data
####
.. doxygenfunction:: dragon::Workspace::data(int64_t size, const string &name = "BufferShared")

graphs
######
.. doxygenfunction:: dragon::Workspace::graphs

name
####
.. doxygenfunction:: dragon::Workspace::name

tensors
#######
.. doxygenfunction:: dragon::Workspace::tensors

Public Functions
----------------

Clear
#####
.. doxygenfunction:: dragon::Workspace::Clear

CreateGraph
###########
.. doxygenfunction:: dragon::Workspace::CreateGraph

CreateTensor
############
.. doxygenfunction:: dragon::Workspace::CreateTensor

GetTensor
#########
.. doxygenfunction:: dragon::Workspace::GetTensor

HasTensor
#########
.. doxygenfunction:: dragon::Workspace::HasTensor

MergeFrom
#########
.. doxygenfunction:: dragon::Workspace::MergeFrom

RunGraph
########
.. doxygenfunction:: dragon::Workspace::RunGraph

RunOperator
###########
.. doxygenfunction:: dragon::Workspace::RunOperator

SetAlias
########
.. doxygenfunction:: dragon::Workspace::SetAlias

TryGetTensor
############
.. doxygenfunction:: dragon::Workspace::TryGetTensor

UniqueName
##########
.. doxygenfunction:: dragon::Workspace::UniqueName

.. raw:: html

  <style>
    h1:before {
      content: "dragon::";
      color: #103d3e;
    }
  </style>
