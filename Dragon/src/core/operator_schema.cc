#include "core/operator_schema.h"

namespace dragon {

bool OpSchema::Verify(const OperatorDef& def) const {
    if (ignore_verify_) return true;
    string indicator = "[" + def.name() + ", " + def.type() + "]\n";
    if (def.input_size() < min_input_ || def.input_size() > max_input_) {
        LOG(FATAL) << indicator << "Input size: " << def.input_size()
                   << " is not in range [min=" << min_input_
                   << ", max=" << max_input_ << "]";
    }
    if (def.output_size() < min_output_ || def.output_size() > max_output_) {
        LOG(FATAL) << indicator << "Output size: " << def.output_size()
                   << " is not in range [min=" << min_output_
                   << ", max=" << max_output_ << "]";
    }
    for (int in = 0; in < def.input_size(); in++) {
        if (def.input(in) == "ignore") continue;
        for (int out = 0; out < def.output_size(); out++) {
            if (def.output(out) == "ignore") continue;
            if (def.input(in) == def.output(out) && (!CheckInplace(in, out)))
                LOG(FATAL) << indicator << "Input("  << in << ") and "
                           << "Output(" << out << ") can not be set to inplace.";
        }
    }
    return true;
}

OpSchema& OpSchema::NumInputs(int min_num, int max_num) {
    min_input_ = min_num;
    max_input_ = max_num;
    return *this;
}

OpSchema& OpSchema::NumInputs(int n) {
    return NumInputs(n, n);
}

OpSchema& OpSchema::NumOutputs(int min_num, int max_num) {
    min_output_ = min_num;
    max_output_ = max_num;
    return *this;
}

OpSchema& OpSchema::NumOutputs(int n) {
    return NumOutputs(n, n);
}

OpSchema& OpSchema::Inplace(set< pair<int, int> > inplace) {
    CheckInplace = [inplace](int in, int out)->bool {
        return (inplace.count(std::make_pair(in, out)) > 0);
    };
    allow_inplace_ = true;
    return *this;
}

}    // namespace dragon