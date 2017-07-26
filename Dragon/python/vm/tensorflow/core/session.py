# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

__all__ = ['Session']

import dragon.core.workspace as ws
import dragon.vm.theano as theano
from dragon.core.tensor import Tensor
from dragon.vm.tensorflow.utils.check import feed_check
from dragon.vm.tensorflow.training.optimizer import BaseOptimizer

TRANSACTIONS = {}

class Transaction(object):
    def __init__(self, functions):
        self.functions = functions

    def run(self, feed_values=None):
        for i, function in enumerate(self.functions):
            # process feeds only for first function
            if i == 0 and feed_values is not None:
                function(*feed_values, return_outputs=False)
            else: function(return_outputs=False)

class Session(object):
    def __init__(self): pass

    def run(self, fetches, feed_dict=None):
        if not isinstance(fetches, list): fetches = [fetches]

        # unpack opts and tensors
        opts = []; tensors = []
        for target in fetches:
            if isinstance(target, BaseOptimizer): opts.append(target)
            elif isinstance(target, Tensor): tensors.append(target)

        # find minimum solving targets
        targets = set()
        for t in tensors: targets.add(t)
        for opt in opts:
            for t in opt.objs: targets.add(t)
        targets = list(targets)

        # if existing a transaction before ?
        global TRANSACTIONS
        t_key = tuple(fetches + feed_dict.keys()) \
                if feed_dict is not None else tuple(fetches)
        transaction = None if not t_key in TRANSACTIONS else TRANSACTIONS[t_key]

        # run through feed
        if feed_dict is not None:
            feed_check(feed_dict)  # check feeds
            if transaction is None: # create a new transaction
                functions = []
                functions.append(theano.function(inputs=feed_dict.keys(), outputs=targets))
                for opt in opts:
                    functions.append(theano.function(updater=opt.updater))
                TRANSACTIONS[t_key] = transaction = Transaction(functions)

            transaction.run(feed_dict.values())

        # run without feed
        else:
            if transaction is None: # create a new transaction
                functions = []
                functions.append(theano.function(outputs=targets))
                for opt in opts:
                    functions.append(theano.function(updater=opt.updater))
                TRANSACTIONS[t_key] = transaction = Transaction(functions)
            transaction.run(None) # run

        # fetch
        rets = []
        for target in fetches:
            if isinstance(target, BaseOptimizer): rets.append(None)
            else:
                __ndarray__ = ws.FetchTensor(target)
                if __ndarray__.size == 1: rets.append(__ndarray__.flatten()[0])
                else: rets.append(__ndarray__)

        if len(rets) == 1: return rets[0]
        else: return rets






