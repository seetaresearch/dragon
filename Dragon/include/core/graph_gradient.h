// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_CORE_GRAPH_GRADIENT_H_
#define DRAGON_CORE_GRAPH_GRADIENT_H_

#include "core/common.h"

namespace dragon {

typedef pair<bool, vector<pair<string, int> > > CheckTuple;

class GraphGradientMaker {
 public:
    GraphGradientMaker(const GraphDef& forward_def, 
                       const vector<string>& targets) 
        : cur_op_idx_(0),
          forward_def_(forward_def) {
        for (auto& target : targets) targets_set_.insert(target);
    }

    GraphDef Make();

    inline void SetTerms(const Map<string, string>& terms) { terms_ = terms; }
    inline void SetOperatorPrefix(const string& prefix) { op_prefix_ = prefix; }
    inline void SetOperatorSuffix(const string& suffix) { op_suffix_ = suffix; }
    inline void AddExternalGrad(const string& name) { external_grads_.insert(name); }

 private:
    CheckTuple CheckMissingGrad(OperatorDef* forward_op);
    string GetOperatorName();

    GraphDef forward_def_, new_def_;
    Map<string, string> terms_, inputs_to_grads_;
    Set<string> targets_set_, blacklist_set_, external_grads_;
    string op_prefix_, op_suffix_;
    int cur_op_idx_;
};

}    // namespace dragon

#endif