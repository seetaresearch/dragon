=============
:mod:`Vision`
=============

.. toctree::
   :hidden:

.. automodule:: dragon.operators.vision
    :members:

.. |conv_output_dim| mathmacro:: \\ DilatedKernelSize = Dilation * (KernelSize - 1) + 1 \\
                                    OutputDim = (InputDim + 2 * Pad - DilatedKernelSize) / Stride + 1

.. |deconv_output_dim| mathmacro:: \\ DilatedKernelSize = Dilation * (KernelSize - 1) + 1 \\
                                    OutputDim = Stride * (InputDim - 1) + DilatedKernelSize - 2 * Pad

.. |pooling_output_dim| mathmacro::  \\ OutputDim = Ceil((InputDim + 2 * Pad - KernelSize) / Stride) + 1