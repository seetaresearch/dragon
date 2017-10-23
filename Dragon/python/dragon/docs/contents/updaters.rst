======================
:mod:`dragon.updaters`
======================

.. toctree::
   :hidden:

Quick Shortcut
--------------

====================    =============================================================================
List                    Brief
====================    =============================================================================
`SGDUpdater`_           The Momentum-SGD Updater, introduced by `[LeCun et.al, 1998]`_.
`NesterovUpdater`_      The Nesterov-SGD Updater, introduced by `[Sutskever et.al, 2012]`_.
`RMSPropUpdater`_       The RMSProp Updater, introduced by `[Hinton et.al, 2013]`_.
`AdamUpdater`_          The Adam Updater, introduced by `[Kingma & Ba, 2014]`_.
====================    =============================================================================

API Reference
-------------

.. currentmodule:: dragon.updaters

.. autoclass:: BaseUpdater
    :members:

    .. automethod:: __init__


.. autoclass:: SGDUpdater
    :members:

    .. automethod:: __init__


.. autoclass:: NesterovUpdater
    :members:

    .. automethod:: __init__


.. autoclass:: RMSPropUpdater
    :members:

    .. automethod:: __init__


.. autoclass:: AdamUpdater
    :members:

    .. automethod:: __init__


.. _[LeCun et.al, 1998]: http://yann.lecun.com/exdb/publis/#lecun-98b
.. _[Sutskever et.al, 2012]: http://www.cs.utoronto.ca/~bonner/courses/2016s/csc321/lectures/lec6.pdf
.. _[Hinton et.al, 2013]: http://www.cs.utoronto.ca/~bonner/courses/2016s/csc321/lectures/lec6.pdf
.. _[Kingma & Ba, 2014]: https://arxiv.org/abs/1412.6980

.. _SGDUpdater: #dragon.updaters.SGDUpdater
.. _NesterovUpdater: #dragon.updaters.NesterovUpdater
.. _RMSPropUpdater: #dragon.updaters.RMSPropUpdater
.. _AdamUpdater:  #dragon.updaters.AdamUpdater