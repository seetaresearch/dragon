OpSchema
========

.. doxygenclass:: dragon::OpSchema

Constructors
------------

.. doxygenfunction:: dragon::OpSchema::OpSchema()
.. doxygenfunction:: dragon::OpSchema::OpSchema(const string &op_type, const string &file, const int line)

Public Functions
----------------

AllowInplace
############
.. doxygenfunction:: dragon::OpSchema::AllowInplace(set<pair<int, int>> inplace)

AllowInplace
############
.. doxygenfunction:: dragon::OpSchema::AllowInplace(std::function<bool(int, int)> inplace)

NumInputs
#########
.. doxygenfunction:: dragon::OpSchema::NumInputs(int n)

NumInputs
#########
.. doxygenfunction:: dragon::OpSchema::NumInputs(int min_num, int max_num)

NumOutputs
##########
.. doxygenfunction:: dragon::OpSchema::NumOutputs(int n)

NumOutputs
##########
.. doxygenfunction:: dragon::OpSchema::NumOutputs(int min_num, int max_num)

Verify
######
.. doxygenfunction:: dragon::OpSchema::Verify

.. raw:: html

  <style>
    h1:before {
      content: "dragon::";
      color: #103d3e;
    }
  </style>
