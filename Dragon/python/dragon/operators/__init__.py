# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor

INT_MAX = 2147483647

def CheckInputs(inputs, *args):
    def Verify(inputs, min_num, max_num):
        # type checking
        if min_num == max_num and min_num == 1:
            if not isinstance(inputs, Tensor):
                raise ValueError('The type of inputs should be Tensor.')
        else:
            if not isinstance(inputs, list) or \
                     not all([isinstance(input, Tensor) for input in inputs]):
                raise ValueError('The type of inputs should be list of Tensor.')
        # range checking
        if not isinstance(inputs, list): inputs = [inputs]
        if len(inputs) < min_num or len(inputs) > max_num:
            raise ValueError('The inputs size is not in range: '
                             '[min={}, max={}].'.format(min_num, max_num))
    args = list(args)
    if len(args) == 1:  # EQ
        if args[0] is None: return  # skip
        else: Verify(inputs, args[0], args[0])
    elif len(args) == 2: # LE and GE
        assert args[0] is not None
        assert args[1] is not None
        Verify(inputs, args[0], args[1])


def ParseArguments(locals):
    __all__ = locals
    kwargs = __all__['kwargs']; del __all__['kwargs']
    return dict(__all__, **kwargs)