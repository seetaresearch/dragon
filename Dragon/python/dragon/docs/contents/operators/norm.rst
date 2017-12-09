====================
:mod:`Normalization`
====================

.. toctree::
   :hidden:

.. automodule:: dragon.operators.norm
    :members:

.. |batchnorm_function| mathmacro:: \\ \, \\ \mu_{B} = \frac{1}{m} \sum_{i=1}^{m}x_{i} \\
                                             \sigma_{B}^{2} = \frac{1}{m} \sum_{i=1}^{m}(x_{i} - \mu_{B})^{2} \\
                                             \hat{x}_{i} = \frac{x_{i} - \mu_{B}}{\sqrt{\sigma_{B}^{2} + \epsilon}} \\ \,

.. |batchnorm_scale_function| mathmacro:: \\ \, \\ \mu_{B} = \frac{1}{m} \sum_{i=1}^{m}x_{i} \\
                                             \sigma_{B}^{2} = \frac{1}{m} \sum_{i=1}^{m}(x_{i} - \mu_{B})^{2} \\
                                             \hat{x}_{i} = \frac{x_{i} - \mu_{B}}{\sqrt{\sigma_{B}^{2} + \epsilon}} \\ y_{i} = \gamma\hat{x}_{i} + \beta \\ \,

.. |batchrenorm_function| mathmacro:: \\ \, \\ \mu_{B} = \frac{1}{m} \sum_{i=1}^{m}x_{i} \\
                                             \sigma_{B}^{2} = \frac{1}{m} \sum_{i=1}^{m}(x_{i} - \mu_{B})^{2} \\
                                             \hat{x}_{i} = \frac{x_{i} - \mu_{B}}{\sqrt{\sigma_{B}^{2} + \epsilon}} \cdot r + d \\ \,

.. |default_moving_average_function| mathmacro:: \\ \, \\ x_{moving} \leftarrow Momentum * x_{moving} + (1 - Momentum) * x_{stat} \\ \,

.. |caffe_moving_average_function| mathmacro:: \\ \, \\ x_{moving} \leftarrow Momentum * x_{moving} + x_{stat} \\ \,


.. _ops.Scale(*args, **kwargs): arithmetic.html#dragon.operators.arithmetic.Scale

.. _Caffe: https://github.com/BVLC/caffe/
