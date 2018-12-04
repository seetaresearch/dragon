/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_GRAPH_GRADIENT_H_
#define DRAGON_CORE_GRAPH_GRADIENT_H_

#include "core/common.h"

namespace dragon {

class GraphGradientMaker {
 public:
    GraphGradientMaker(): cur_op_idx_(0) {}

    void Make(
        const GraphDef&         forward_def,
        const vector<string>&   targets,
        GraphDef&               new_def);

    void Share(const string& grads_prefix, GraphDef& graph);

    inline void SetTerms(
        const Map<string, string>& terms) { terms_ = terms; }
    inline void SetOperatorPrefix(
        const string& prefix) { op_prefix_ = prefix; }
    inline void SetOperatorSuffix(
        const string& suffix) { op_suffix_ = suffix; }
    inline void AddExternalGrad(
        const string& name) { external_grads_.insert(name); }
    inline void AddIgnoreGrad(
        const string& name) { ignore_grads_.insert(name); }

 private:
    bool CheckGrad(
        const OperatorDef&              forward_op,
        const Set<string>&              targets,
        vector< pair<string, int> >&    gen_grads);

    string GetOperatorName();

    Map<string, string> terms_, inputs_to_grads_;
    Set<string> blacklist_set_, external_grads_, ignore_grads_;
    string op_prefix_, op_suffix_;
    int cur_op_idx_;
};

}  // namespace dragon

#endif  // DRAGON_CORE_GRAPH_GRADIENT_H_