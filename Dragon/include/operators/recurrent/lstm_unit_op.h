// -// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_RECURRENT_LSTM_UNIT_OP_H_
#define DRAGON_OPERATORS_RECURRENT_LSTM_UNIT_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class LSTMUnitOp : public Operator<Context> {
 public:
    LSTMUnitOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          has_cont(OperatorBase::GetSingleArg<string>("cont_t", "")) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex num, channels;
    string has_cont;
    Tensor* cont_t;
};

template <class Context>
class LSTMUnitGradientOp : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(LSTMUnitGradientOp);

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex num, channels;
    Tensor* zeros;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_RECURRENT_LSTM_UNIT_OP_H_