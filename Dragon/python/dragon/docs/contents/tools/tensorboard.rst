==================
:mod:`TensorBoard`
==================

.. toctree::
   :hidden:

Quick Shortcut
--------------

====================    =============================================================================
List                    Brief
====================    =============================================================================
`scalar_summary`_       Write a scalar variable.
`histogram_summary`_    Write a histogram of values.
`image_summary`_        Write a list of images.
`close`_                Close the board and apply all cached summaries.
====================    =============================================================================


API Reference
-------------

.. currentmodule:: dragon.tools.tensorboard

.. autoclass:: TensorBoard
    :members:

    .. automethod:: __init__

.. _scalar_summary: tensorboard.html#dragon.tools.tensorboard.TensorBoard.scalar_summary
.. _histogram_summary: tensorboard.html#dragon.tools.tensorboard.TensorBoard.histogram_summary
.. _image_summary: tensorboard.html#dragon.tools.tensorboard.TensorBoard.image_summary
.. _close: tensorboard.html#dragon.tools.tensorboard.TensorBoard.close

