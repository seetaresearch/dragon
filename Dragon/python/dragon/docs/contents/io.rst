================
:mod:`dragon.io`
================

Wrapper
-------

.. toctree::
   :hidden:

   io/data_batch

==========================      =====================================================================
List                            Brief
==========================      =====================================================================
`dragon.io.data_batch`_         Efficient I/O based on `LMDB`_.
==========================      =====================================================================

Component
---------

.. toctree::
   :hidden:

   io/data_reader
   io/data_transformer
   io/blob_fetcher

==============================      =====================================================================
List                                Brief
==============================      =====================================================================
`dragon.io.data_reader`_            Queue encoded string from `LMDB`_.
`dragon.io.data_transformer`_       Queue transformed images from `DataReader`_.
`dragon.io.blob_fetcher`_           Queue blobs from `DataTransformer`_.
==============================      =====================================================================


.. _LMDB: http://lmdb.readthedocs.io/en/release
.. _DataReader: io/data_reader.html#dragon.io.data_reader
.. _DataTransformer: io/data_transformer.html#dragon.io.data_transformer
.. _dragon.io.data_batch: io/data_batch.html
.. _dragon.io.data_reader: io/data_reader.html
.. _dragon.io.data_transformer: io/data_transformer.html
.. _dragon.io.blob_fetcher: io/blob_fetcher.html