# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor
import dragon.core.mpi as mpi

def MPIBroadcast(inputs, root, mpi_rank=None, **kwargs):
    """
    :param inputs:          a Tensor which to broadcast
    :param root:            a int of the root in a broadcast group
    :return:                a Tensor that be broadcast
    """

    if not isinstance(inputs, Tensor):
        raise RuntimeError('MPIBroadcast Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    if mpi_rank is None:
        num_nodes = mpi.size()
        mpi_rank = [i for i in xrange(0, num_nodes)]
    if not isinstance(kwargs['mpi_rank'], list):
        kwargs['mpi_rank'] = [kwargs['mpi_rank']]

    comm, group = mpi.group(root, incl=mpi_rank)
    new_kwargs = {'inputs': kwargs['inputs'], 'mpi_rank': mpi_rank,
                  'comm': comm, 'group': group}

    return Tensor.CreateOperator(nout=1, op_type='MPIBroadcast', **new_kwargs)


def MPIGather(inputs, root, mpi_rank=None, **kwargs):
    """
    :param inputs:          a Tensor which to broadcast
    :param root:            a int of the root in a broadcast group
    :return:                a Tensor that be broadcast
    """

    if not isinstance(inputs, Tensor):
        raise RuntimeError('MPIGather Operator accepts a Tensor as inputs')

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    if mpi_rank is None:
        num_nodes = mpi.size()
        kwargs['mpi_rank'] = [i for i in xrange(0, num_nodes)]
    if not isinstance(kwargs['mpi_rank'], list):
        kwargs['mpi_rank'] = [kwargs['mpi_rank']]

    if kwargs.has_key('nout'):
        if kwargs['nout'] != len(kwargs['mpi_rank']):
            raise RuntimeError('specfied nout is {}, but provide {} mpi nodes'
                               .format(kwargs['nout'], len(kwargs['mpi_rank'])))
        safe_nout = kwargs['nout']
    else: safe_nout = len(kwargs['mpi_rank'])

    comm, group = mpi.group(root, incl=mpi_rank)
    new_kwargs = {'inputs': kwargs['inputs'], 'mpi_rank': kwargs['mpi_rank'],
                  'comm': comm, 'group': group}

    return Tensor.CreateOperator(nout=safe_nout, op_type='MPIGather', **new_kwargs)