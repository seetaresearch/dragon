# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

COLLECTIONS = {}

__all__ = [
    'add_to_collection',
    'get_collection'
]

def add_to_collection(name, value):
    global COLLECTIONS
    if not name in COLLECTIONS:
        COLLECTIONS[name] = []
    COLLECTIONS[name].append(value)

def get_collection(key, scope=None):
    collection = COLLECTIONS[key]
    if scope is None: return collection

    filter_collection = []
    for value in collection:
        if hasattr(value, 'name'):
            if value.name.startwith(scope):
                filter_collection.append(value)
        else: filter_collection.append(value)

    return filter_collection
