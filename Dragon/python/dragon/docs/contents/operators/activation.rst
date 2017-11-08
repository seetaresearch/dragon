=================
:mod:`Activation`
=================

.. toctree::
   :hidden:

.. automodule:: dragon.operators.activation
    :members:

.. |sigmoid_function| mathmacro:: \, y = \frac{1}{1 + {e}^{-x}}

.. |tanh_function| mathmacro:: \, y = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

.. |relu_function| mathmacro:: \, y = \left\{ \begin{array} \\ x & & (x > 0) \\ 0 & & (x <= 0) \\ \end{array} \right.

.. |lrelu_function| mathmacro::  \, y = \left\{ \begin{array} \\ x & & (x > 0) \\ Slope * x & & (x <= 0) \\ \end{array} \right.

.. |prelu_function| mathmacro:: \, y_{i} = \left\{ \begin{array} \\ x_{i} & & (x_{i} > 0) \\ \alpha_{i} * x_{i} & & (x <= 0) \\ \end{array} \right.

.. |elu_function| mathmacro:: \, y = \left\{ \begin{array} \\ x & & (x > 0) \\ Alpha * (e^{x} - 1) & & (x <= 0) \\ \end{array} \right.

.. |selu_function| mathmacro:: \, y = 1.0507 \left\{ \begin{array} \\ x & & (x > 0) \\ 1.6733 * (e^{x} - 1) & & (x <= 0) \\ \end{array} \right.

.. |dropout_function| mathmacro:: \, y = x * Bernoulli(p=1 - prob)

.. |softmax_function| mathmacro:: \, y = \frac{e^{x_{i}}}{\sum  e^{x_{j}}}
