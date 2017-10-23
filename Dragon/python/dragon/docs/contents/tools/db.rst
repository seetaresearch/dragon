===========
:mod:`LMDB`
===========

.. toctree::
   :hidden:

Quick Shortcut
--------------

====================    =============================================================================
List                    Brief
====================    =============================================================================
`LMDB.open`_            Open the database.
`LMDB.put`_             Put the item.
`LMDB.commit`_          Commit all items that have been put.
`LMDB.set`_             Set the cursor to the specific key.
`LMDB.get`_             Get the value of the specific key.
`LMDB.next`_            Set the cursor to the next.
`LMDB.key`_             Get the key under the current cursor.
`LMDB.value`_           Get the value under the current cursor.
`LMDB.close`_           Close the database.
====================    =============================================================================

API Reference
-------------

.. currentmodule:: dragon.tools.db

.. autoclass:: LMDB
    :members:

    .. automethod:: __init__

.. _LMDB.open: #dragon.tools.db.LMDB.open
.. _LMDB.put: #dragon.tools.db.LMDB.put
.. _LMDB.commit: #dragon.tools.db.LMDB.commit
.. _LMDB.set: #dragon.tools.db.LMDB.set
.. _LMDB.get: #dragon.tools.db.LMDB.get
.. _LMDB.next: #dragon.tools.db.LMDB.next
.. _LMDB.key: #dragon.tools.db.LMDB.key
.. _LMDB.value: #dragon.tools.db.LMDB.value
.. _LMDB.close: #dragon.tools.db.LMDB.close