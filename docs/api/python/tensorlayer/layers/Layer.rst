Layer
=====

.. autoclass:: dragon.vm.tensorlayer.layers.Layer

__init__
--------
.. automethod:: dragon.vm.tensorlayer.layers.Layer.__init__

Properties
----------

all_weights
###########
.. autoattribute:: dragon.vm.tensorlayer.layers.Layer.all_weights

name
####
.. autoattribute:: dragon.vm.tensorlayer.layers.Layer.name

nontrainable_weights
####################
.. autoattribute:: dragon.vm.tensorlayer.layers.Layer.nontrainable_weights

trainable_weights
#################
.. autoattribute:: dragon.vm.tensorlayer.layers.Layer.trainable_weights

training
########
.. autoattribute:: dragon.vm.tensorlayer.layers.Layer.training

Methods
-------

add_weight
##########
.. automethod:: dragon.vm.tensorlayer.Module.add_weight
  :noindex:

build
#####
.. automethod:: dragon.vm.tensorlayer.layers.Layer.build

forward
#######
.. automethod:: dragon.vm.tensorlayer.layers.Layer.forward

load_weights
############
.. automethod:: dragon.vm.tensorlayer.Module.load_weights
  :noindex:

save_weights
############
.. automethod:: dragon.vm.tensorlayer.Module.save_weights
  :noindex:

.. raw:: html

  <style>
    h1:before {
      content: "tl.layers.";
      color: #103d3e;
    }
  </style>
