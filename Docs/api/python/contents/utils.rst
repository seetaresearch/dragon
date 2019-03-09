===================
:mod:`dragon.utils`
===================

Vision
------

.. toctree::
   :hidden:

   utils/vision/database
   utils/vision/data_batch
   utils/vision/data_reader
   utils/vision/data_transformer
   utils/vision/blob_fetcher

=========================================    =====================================================================
List                                         Brief
=========================================    =====================================================================
`dragon.utils.vision.im2db`_                 Make the sequential database for images.
`dragon.utils.vision.data_batch`_            Efficient Batch data provider based on `LMDB`_.
`dragon.utils.vision.data_reader`_           Queue encoded string from `LMDB`_.
`dragon.utils.vision.data_transformer`_      Queue transformed images from `DataReader`_.
`dragon.utils.vision.blob_fetcher`_          Queue blobs from `DataTransformer`_.
=========================================    =====================================================================

.. _LMDB: http://lmdb.readthedocs.io/en/release
.. _dragon.utils.vision.im2db: utils/vision/database.html
.. _DataReader: utils/vision/data_reader.html#dragon.utils.vision.data_reader
.. _DataTransformer: utils/vision/data_transformer.html#dragon.utils.vision.data_transformer
.. _dragon.utils.vision.data_batch: utils/vision/data_batch.html
.. _dragon.utils.vision.data_reader: utils/vision/data_reader.html
.. _dragon.utils.vision.data_transformer: utils/vision/data_transformer.html
.. _dragon.utils.vision.blob_fetcher: utils/vision/blob_fetcher.html