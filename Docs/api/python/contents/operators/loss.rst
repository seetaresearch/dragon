===========
:mod:`Loss`
===========

.. toctree::
   :hidden:

.. automodule:: dragon.operators.loss
    :members:

.. |smooth_l1_beta| mathmacro:: \, \frac{1}{\sigma^{2}}

.. |l1_loss_function| mathmacro:: \, Loss = \frac{ \sum \left|  Weight * (Input - Target) \right|}{ Normalization}

.. |l2_loss_function| mathmacro:: \, Loss = \frac{ \sum \frac{1}{2}\left|\left|  Weight * (Input - Target) \right|\right|}{ Normalization}