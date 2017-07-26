# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor

def LSTMUnit(c_t_1, gate_input, cont_t=None, **kwargs):
    kwargs = {}
    if cont_t is not None:
        if isinstance(cont_t, Tensor):
            raise TypeError('cont_t must be a Tensor')
        kwargs['cont_t'] = cont_t.name
    return Tensor.CreateOperator(
        inputs=[c_t_1, gate_input], nout=2, op_type='LSTMUnit', **kwargs)










