===================
:mod:`dragon.tools`
===================

Overview
--------

|para| In this section, we will introduce several tools for flexible tasks of Deep Learning.

|para| Those tools are based on the existing python packages,
which are easy to install(by `pip`_) and can adapt to all os platforms. We will never rebuild them
in the C++ backend because they could make our kernels dirty and messy(especially the **Sequential Databases**).


ToolBox
-------

.. toctree::
   :hidden:

   tools/db
   tools/im2db
   tools/summary_writer

====================    ====================================================================================
List                    Brief
====================    ====================================================================================
`LMDB`_                 A wrapper of LMDB package.
`IM2DB`_                Make the sequential database for images.
`SummaryWriter`_        Write summaries for DragonBoard.
====================    ====================================================================================


.. |para| raw:: html

    <p style="text-indent:1.5em; font-size: 18px; max-width: 830px;">

.. _pip: https://pypi.python.org/pypi/pip

.. _LMDB: tools/db.html
.. _IM2DB: tools/im2db.html
.. _SummaryWriter: tools/summary_writer.html
