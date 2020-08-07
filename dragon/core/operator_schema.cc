#include "dragon/core/operator_schema.h"

namespace dragon {

bool OpSchema::Verify(const OperatorDef& def) const {
  auto header = "[" + def.name() + ", " + def.type() + "]\n";
  if (def.input_size() < min_input_ || def.input_size() > max_input_) {
    LOG(FATAL) << header << "Input size: " << def.input_size()
               << " is not in range [min=" << min_input_
               << ", max=" << max_input_ << "]";
  }
  if (def.output_size() < min_output_ || def.output_size() > max_output_) {
    LOG(FATAL) << header << "Output size: " << def.output_size()
               << " is not in range [min=" << min_output_
               << ", max=" << max_output_ << "]";
  }
  for (int i = 0; i < def.input_size(); ++i) {
    if (def.input(i).empty()) continue;
    for (int j = 0; j < def.output_size(); ++j) {
      if (def.output(j).empty()) continue;
      if (def.input(i) == def.output(j) && !CheckInplace(i, j)) {
        LOG(FATAL) << header << "Input(" << i << ") and Output(" << j << ") "
                   << "can not be set to inplace.";
      }
    }
  }
  return true;
}

OpSchema& OpSchema::NumInputs(int n) {
  return NumInputs(n, n);
}

OpSchema& OpSchema::NumOutputs(int n) {
  return NumOutputs(n, n);
}

OpSchema& OpSchema::NumInputs(int min_num, int max_num) {
  min_input_ = min_num;
  max_input_ = max_num;
  return *this;
}

OpSchema& OpSchema::NumOutputs(int min_num, int max_num) {
  min_output_ = min_num;
  max_output_ = max_num;
  return *this;
}

OpSchema& OpSchema::AllowInplace(std::function<bool(int, int)> inplace) {
  CheckInplace = inplace;
  return *this;
}

OpSchema& OpSchema::AllowInplace(set<pair<int, int>> inplace) {
  CheckInplace = [inplace](int in, int out) -> bool {
    return (inplace.count(std::make_pair(in, out)) > 0);
  };
  return *this;
}

Map<string, OpSchema>& OpSchemaRegistry::schema_map() {
  static Map<string, OpSchema> schema_map_;
  return schema_map_;
}

} // namespace dragon
