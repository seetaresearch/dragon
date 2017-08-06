#include "core/graph_gradient.h"
#include "core/operator.h"
#include "utils/string.h"

namespace dragon {

#define str dragon_cast<std::string, int>

CheckTuple GraphGradientMaker::CheckMissingGrad(OperatorDef* forward_op) {
    vector< pair<string, int> > gen_grads;
    if (NoGradientRegistry()->Has(forward_op->type())) {
        for (auto& input : forward_op->input()) blacklist_set_.insert(input);
        return { true, gen_grads };
    }
    for (int idx = 0; idx < forward_op->output_size(); idx++) {
        string output = forward_op->output(idx);
        if (!inputs_to_grads_.count(output)) {
            string g_output = output + "_grad";
            if (terms_.count(g_output)) g_output = terms_[g_output];
    
            //  check if having external grad first
            if (external_grads_.count(g_output))
                inputs_to_grads_[output] = g_output;

            //  consider generate virtual grad
            else if (targets_set_.count(output) && g_output != "ignore") {
                gen_grads.push_back({ output, idx });
                inputs_to_grads_[output] = g_output;
            }
        }
    }
    //  blacklist check
    for (auto& output : forward_op->output()) {
        if (!inputs_to_grads_.count(output)) {
            if (blacklist_set_.count(output)) return { true, gen_grads };
            if (forward_op->output_size() == 1) return { true, gen_grads };
        }
    }
    //  check pass, even if missing some grads
    return { false, gen_grads };
}

string GraphGradientMaker::GetOperatorName() {
    return op_prefix_ + str(cur_op_idx_++) + op_suffix_;
}

GraphDef GraphGradientMaker::Make() {
    CHECK(!op_prefix_.empty()) << "please set a prefix before making.";
    Map<string, int> inputs_count, grads_count;
    Set<string> all_split_grads;

    // PLAY for the forward
    for (auto& op : forward_def_.op()) {
        if (NoGradientRegistry()->Has(op.type())) continue;
        for (auto& input : op.input()) inputs_count[input]++;
    }

    // PLAY for the backward
    for (int i = forward_def_.op_size() - 1; i >= 0; i--) {
        vector<string> outputs, g_outputs;
        OperatorDef* op = forward_def_.mutable_op(i);
        for (auto& output : op->output()) outputs.push_back(output);
        CheckTuple tuple = CheckMissingGrad(op);
        bool is_skip = tuple.first;
        vector< pair<string, int> > gen_grads = tuple.second;

        for (auto& output : outputs) {
            if (inputs_to_grads_.count(output))
                g_outputs.push_back(inputs_to_grads_[output]);
            else g_outputs.push_back("ignore");
        }

        Gradient grad = MakeGradientForOp(*op, g_outputs);

        // replace terms
        for (auto& g_op : grad.ops) {
            g_op.set_name(GetOperatorName());
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
        OperatorDef* gather_op = nullptr;

        for (auto& g_op : grad.ops) {
            for (int i = 0; i < g_op.output_size(); i++) {
                string* output = g_op.mutable_output(i);
                int original_idx = -1;
                for (int j = 0; j < grad.g_inputs.size(); j++) 
                    if (g_op.output(i) == grad.g_inputs[j]) original_idx = j;
                if (original_idx == -1) continue;
                string original_name = op->input(original_idx);

                if (inputs_count[original_name] > 1) {
                    //  split
                    string split_name = *output + "_autosplit_" + str(grads_count[*output]++);
                    if (!is_skip) all_split_grads.insert(split_name);
                    //  gather
                    if (grads_count[*output] == inputs_count[original_name]) {
                        gather_op = new OperatorDef();
                        gather_op->set_name(GetOperatorName());
                        gather_op->set_type("GradientGather");
                        gather_op->add_output(*output);
                        if (g_op.has_device_option())
                            gather_op->mutable_device_option()->CopyFrom(g_op.device_option());
                        for (int j = 0; j < grads_count[*output]; j++) {
                            string key = *output + "_autosplit_" + str(j);
                            if (all_split_grads.count(key)) gather_op->add_input(key);
                        }
                    }
                    *output = split_name;
                }
            }
        }

        //  append ops
        if (!is_skip) {
            if (gen_grads.size() > 0) {
                vector<string> op_inputs, op_outputs;
                Argument arg_defaults; arg_defaults.set_name("defaults");
                for (auto& gen_grad : gen_grads) {
                    op_inputs.push_back(gen_grad.first);
                    string output = gen_grad.first + "_grad";
                    if (terms_.count(output)) output = terms_[output];
                    op_outputs.push_back(output);
                    arg_defaults.add_floats(grad.defaults[gen_grad.second]);
                }
                OperatorDef generate_op = MakeOperatorDef("GradientGenerate", 
                                                           GetOperatorName(),
                                                                   op_inputs, 
                                                                  op_outputs, 
                                          vector<Argument>(1, arg_defaults));
                if (op->has_device_option())
                    generate_op.mutable_device_option()->CopyFrom(op->device_option());
                new_def_.add_op()->CopyFrom(generate_op);
            }
            for (auto& g_op : grad.ops) new_def_.add_op()->CopyFrom(g_op);
        }
        if (gather_op != nullptr) new_def_.add_op()->CopyFrom(*gather_op);

        //  done
        if (!is_skip) {
            for (int i = 0; i < op->input_size(); i++) {
                if (!grad.g_inputs[i].empty())
                    inputs_to_grads_[op->input(i)] = grad.g_inputs[i];
            }
        }
    }
    forward_def_.MergeFrom(new_def_);
    return forward_def_;
}

}    // namespace dragon