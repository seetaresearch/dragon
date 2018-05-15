// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_CONTROL_FLOW_SCAN_OP_H_
#define DRAGON_OPERATORS_CONTROL_FLOW_SCAN_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ScanOp final: public Operator<Context> {
 public:
    ScanOp(const OperatorDef& op_def, Workspace *ws) 
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 0)),
          nsteps(OperatorBase::GetSingleArg<int>("nsteps", 0)),
          step_type(OperatorBase::GetSingleArg<string>("step_type", "Static")),
          step_tensor(OperatorBase::GetSingleArg<string>("step_tensor", "")),
          nseqs(OperatorBase::GetSingleArg<int>("nseqs", 0)),
          default_outputs(OperatorBase::GetRepeatedArg<string>("default_outputs")),
          nout((int)default_outputs.size()),
          debug_mode(OperatorBase::GetSingleArg<bool>("debug_mode", false)) { 
        InitTemplate(); 
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    void InitTemplate();
    void UnrollTemplate();
    void UpdateTerms(int cur_step);

 protected:
    GraphDef func_def, template_def, new_def;
    Map<int, unique_ptr<Graph>> graphs;
    Graph* cur_graph;
    Map<string, string> terms;
    vector<string> default_outputs;
    TIndex axis, nseqs, nsteps, nrepeats, nout;
    string step_type, step_tensor;
    bool debug_mode;
};

template <class Context>
class ScanGradientOp final: public Operator<Context> {
 public:
    ScanGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", 0)),
          nsteps(OperatorBase::GetSingleArg<int>("nsteps", 0)),
          step_type(OperatorBase::GetSingleArg<string>("step_type", "Static")),
          step_tensor(OperatorBase::GetSingleArg<string>("step_tensor", "")),
          forward_inputs(OperatorBase::GetRepeatedArg<string>("inputs_name")),
          forward_outputs(OperatorBase::GetRepeatedArg<string>("outputs_name")) {
        //  handle GO(x)
        for (int i = 0; i < forward_outputs.size(); i++)
            terms[forward_outputs[i] + "_grad"] = Input(i + (int)OutputSize()).name();
            
        //  handle GI(x)
        for (int i = 0; i < forward_inputs.size(); i++)
            terms[forward_inputs[i] + "_grad"] = Output(i)->name();
    }
    USE_OPERATOR_FUNCTIONS(Context);

    void RunOnDevice() override;
    void MakeOps(const GraphDef& forward_def, GraphDef& new_def);

 protected:
    Map<string, string> terms;
    Map<int, unique_ptr<Graph>> graphs;
    vector<string> forward_inputs, forward_outputs;
    Graph* cur_graph;
    TIndex axis, nsteps;
    string step_type, step_tensor;
};

}    // namespace dragon

#endif    // DRAGON_OPERATORS_CONTROL_FLOW_SCAN_OP_H_