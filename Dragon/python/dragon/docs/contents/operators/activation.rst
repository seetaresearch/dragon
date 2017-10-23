=================
:mod:`Activation`
=================

.. toctree::
   :hidden:

.. automodule:: dragon.operators.activation
    :members:

.. |sigmoid_function| mathmacro:: \, y = \frac{1}{1 + {e}^{-x}}

.. |tanh_function| mathmacro:: \, y = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

.. |relu_function| mathmacro:: \, y = \max(0, x)

.. |elu_function| mathmacro:: \, y = \left\{ \begin{array} \\ x & & (x > 0) \\ Alpha * (e^{x} - 1) & & (x <= 0) \\ \end{array} \right.

.. |leaky_relu_function| mathmacro:: \, y = \max(x, 0) + Slope * \min(x, 0)

.. |dropout_function| mathmacro:: \, y = x * Bernoulli(p=1 - prob)

.. |softmax_function| mathmacro:: \, y = \frac{e^{x_{i}}}{\sum  e^{x_{j}}}
