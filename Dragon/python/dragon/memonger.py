# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

def share_grads(enabled=True):
    from dragon.config import option
    option['share_grads'] = enabled

def drop(op_func, *args, **kwargs):
    kwargs['mirrow_stage'] = True
    return op_func(*args, **kwargs)