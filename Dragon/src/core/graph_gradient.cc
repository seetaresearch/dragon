#include "core/graph_gradient.h"
#include "core/operator.h"
#include "utils/string.h"

namespace dragon {

bool GraphGradientMaker::CheckGrad(
    const OperatorDef&              forward_op,
    const Set<string>&              targets,
    vector< pair<string, int> >&    gen_grads) {
    if (NoGradientRegistry()->Has(forward_op.type())) {
        for (auto& input : forward_op.input()) 
            blacklist_set_.insert(input);
        return true;
    }
    for (int idx = 0; idx < forward_op.output_size(); idx++) {
        string output = forward_op.output(idx);
        if (!inputs_to_grads_.count(output)) {
            string g_output = output + "_grad";
            if (terms_.count(g_output)) g_output = terms_[g_output];
            //  check if having external grad first
            if (external_grads_.count(g_output))
                inputs_to_grads_[output] = g_output;
            //  consider generate virtual grad
            else if (targets.count(output) && g_output != "ignore") {
                gen_grads.push_back({ output, idx });
                inputs_to_grads_[output] = g_output;
            }
        }
    }
    //  blacklist check
    for (auto& output : forward_op.output()) {
        if (!inputs_to_grads_.count(output)) {
            if (blacklist_set_.count(output)) return true;
            if (forward_op.output_size() == 1) return true;
        }
    }
    //  check pass, even if missing some grads
    return false;
}

string GraphGradientMaker::GetOperatorName() {
    if (op_prefix_.empty()) return "runtime";
    return op_prefix_ + std::to_string(cur_op_idx_++) + op_suffix_;
}

void GraphGradientMaker::Make(
    const GraphDef&                 forward_def,
    const vector<string>&           targets,
    GraphDef&                       new_def) {
    Map<string, int> inputs_count, grads_count;
    Set<string> all_split_grads, targets_set;
    //  PLAY for the forward
    for (auto& op : forward_def.op()) {
        if (NoGradientRegistry()->Has(op.type())) continue;
        for (auto& input : op.input()) {
            bool input_in_outputs = false;
            for (auto& output : op.output())
                if (output == input) { input_in_outputs = true; break; }
            // avoid to count the duplicate input(i.e. the in-place output)
            if (!input_in_outputs) inputs_count[input]++;
        }
    }
    for (auto& t : targets) targets_set.insert(t);

    //  PLAY for the backward
    for (int i = forward_def.op_size() - 1; i >= 0; i--) {
        //  collect inputs & outputs, generate RAW grad ops
        const OperatorDef& op = forward_def.op(i);
        vector< pair<string, int> > gen_grads;
        bool is_skip = CheckGrad(op, targets_set, gen_grads);
        vector<string> g_outputs;
        for (auto& output : op.output()) {
            string g_output = "";
            if (inputs_to_grads_.count(output) > 0)
                g_output = inputs_to_grads_[output];
            if (g_output.empty()) g_output = "ignore";
            g_outputs.emplace_back(g_output);
        }
        Gradient grad = MakeGradientForOp(op, g_outputs);

        //  process the RAW grad ops
        vector<OperatorDef> gather_ops;
        for (auto& g_op : grad.ops) {
            //  set op name
            g_op.set_name(GetOperatorName());
            //  rename if necessary
            if (terms_.size() > 0) {
                for (int i = 0; i < g_op.input_size(); i++) {
                    string* input = g_op.mutable_input(i);
                    if (terms_.count(*input)) *input = terms_[*input];
                }
                for (int i = 0; i < g_op.output_size(); i++) {
                    string* output = g_op.mutable_output(i);
                    if (terms_.count(*output)) *output = terms_[*output];
                }
                for (int i = 0; i < grad.g_inputs.size(); i++) {
                    if (terms_.count(grad.g_inputs[i]))
                        grad.g_inputs[i] = terms_[grad.g_inputs[i]];
                }
            }
            //  split & gather grads for multi-used input
            for (int i = 0; i < g_op.output_size(); i++) {
                string* output = g_op.mutable_output(i);
                int original_idx = -1;
                for (int j = 0; j < grad.g_inputs.size(); j++)
                    if (g_op.output(i) == grad.g_inputs[j]) original_idx = j;
                //  ignore un-used && in-placed GI(?)
                if (original_idx == -1) continue;
                bool output_in_inputs = false;
                for (auto& input : g_op.input())
                    if (g_op.output(i) == input) output_in_inputs = true;
                if (output_in_inputs) continue;
                //  found a split branch
                string original_name = op.input(original_idx);
                if (inputs_count[original_name] > 1) {
                    //  split
                    string split_name = *output + "_autosplit_" 
                        + std::to_string(grads_count[*output]++);
                    if (!is_skip) all_split_grads.insert(split_name);
                    //  gather
                    if (grads_count[*output] == inputs_count[original_name]) {
                        OperatorDef gather_op;
                        gather_op.set_name(GetOperatorName());
                        gather_op.set_type("GradientGather");
                        gather_op.add_output(*output);
                        if (g_op.has_device_option())
                            gather_op.mutable_device_option()
                                ->CopyFrom(g_op.device_option());
                        for (int j = 0; j < grads_count[*output]; j++) {
                            string key = *output + "_autosplit_" + std::to_string(j);
                            if (all_split_grads.count(key)) gather_op.add_input(key);
                        }
                        gather_ops.emplace_back(gather_op);
                    }
                    *output = split_name;
                }
            }
        }

        //  now, append the required ops
        if (!is_skip) {
            //  1) GradientGenerateOp
            if (gen_grads.size() > 0) {
                vector<string> op_inputs, op_outputs;
                Argument arg_defaults; arg_defaults.set_name("defaults");
                for (auto& gen_grad : gen_grads) {
                    op_inputs.emplace_back(gen_grad.first);
                    string output = gen_grad.first + "_grad";
                    if (terms_.count(output)) output = terms_[output];
                    op_outputs.emplace_back(output);
                    arg_defaults.add_floats(grad.defaults[gen_grad.second]);
                }
                OperatorDef generate_op = MakeOperatorDef(
                    "GradientGenerate", GetOperatorName(),
                        op_inputs, op_outputs,
                            vector<Argument>(1, arg_defaults));
                if (op.has_device_option())
                    generate_op.mutable_device_option()
                        ->CopyFrom(op.device_option());
                new_def.add_op()->CopyFrom(generate_op);
            }
            //  2) GradientOp
            for (auto& g_op : grad.ops)
                new_def.add_op()->CopyFrom(g_op);
        }
        //  3) GradientGatherOp
        for (auto& gather_op : gather_ops)
            new_def.add_op()->CopyFrom(gather_op);

        //  done!
        if (!is_skip) {
            for (int i = 0; i < op.input_size(); i++) {
                if (!grad.g_inputs[i].empty())
                    inputs_to_grads_[op.input(i)] = grad.g_inputs[i];
            }
        }
    }
}

#define TEMPORARY_GRADS_LIMITS 2
#define SHARE_OUTPUTS_BODY \
   {string output = op->output(ix); \
    if (output == "ignore") continue; \
    if (ref_count.count(output) == 0) { \
        if (ignore_grads_.count(output) > 0) \
            *op->mutable_output(ix) = "ignore"; \
        continue; \
    } \
    if (output.find("autosplit") != string::npos) continue; \
    if (op->type() == "TemplateGradient" || \
        op->type() == "ScanGradient") continue; \
    string temp_grad = output; \
    if (inplace_flags[ix] >= 0) { \
        temp_grad = op->input(inplace_flags[ix]); \
    } else if (grads_pool.size() > 0) { \
        temp_grad = grads_pool.front(); \
        grads_pool.pop_front(); \
        temporary_grads[output] = temp_grad; \
    } \
    *op->mutable_output(ix) = temp_grad;}

void GraphGradientMaker::Share(
    const string&                   grads_prefix,
    GraphDef&                       graph) {
    Map<string, int> ref_count;
    //  count the refs for detecting leaf nodes
    for (auto& op : graph.op()) {
        //  ignore the non-gradient ops
        if (op.type().find("Gradient") == string::npos) continue;
        for (auto& input : op.input())
            if (input.find("grad") != string::npos) ref_count[input] += 1;
    }
    //  heuristically limits the max number of temporal grads
    //  for stable and efficient memory reusing
    //  "2" works well under the most conditions
    //  which represents "GIVE ME ONE, THEN GIVE YOU ONE"
    Map<string, string> temporary_grads;
    std::deque<string> grads_pool; 
    for (int i = 0; i < TEMPORARY_GRADS_LIMITS; i++)
        grads_pool.push_back(grads_prefix + ":" + std::to_string(i));

    for (int i = 0; i < graph.op_size(); i++) {
        OperatorDef* op = graph.mutable_op(i);
        //  ignore the non-gradient ops
        if (op->type().find("Gradient") == string::npos) continue;
        //  GC to store the grads that have finished lifecycle
        vector<string> GC;
        //  inplace-aware
        vector<int> inplace_flags;
        for (int oix = 0; oix < op->output_size(); oix++) {
            int flag = -1;
            for (int iix = 0; iix < op->input_size(); iix++)
                if (op->output(oix) == op->input(iix)) { 
                    flag = iix; break; }
            inplace_flags.emplace_back(flag);
        }
        //  check input grads
        for (int ix = 0; ix < op->input_size(); ix++) {
            string input = op->input(ix);
            if (ref_count.count(input) > 0) {
                ref_count[input] -= 1; // decref
                if (temporary_grads.count(input) == 0) continue;
                string temp_grad = temporary_grads[input];
                // replace it by the temporary grad
                *op->mutable_input(ix) = temp_grad;
                if (ref_count[input] == 0) GC.emplace_back(temp_grad);
            }
        }
        //  determine the scanning order
        bool left = true;
        static Set<string> ROrderOps = {
            "ConcatGradient", "StackGradient",
            "RAddGradient", "RSubGradient", "RMulGradient", "RDivGradient",
        };
        if (ROrderOps.count(op->type())) left = false;
        // check output grads, left order
        for (int ix = 0; ix < op->output_size() && left; ix++) SHARE_OUTPUTS_BODY;
        // check output grads, right order
        for (int ix = op->output_size() - 1; ix >= 0 && !left; ix--) SHARE_OUTPUTS_BODY;
        // update the pool from GC
        for (auto& grads : GC) grads_pool.emplace_back(grads);
    }
}

}    // namespace dragon