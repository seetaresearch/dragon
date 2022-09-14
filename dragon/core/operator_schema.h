/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_OPERATOR_SCHEMA_H_
#define DRAGON_CORE_OPERATOR_SCHEMA_H_

#include "dragon/core/common.h"

namespace dragon {

/*!
 * \brief Class to record the schema of operator.
 */
class DRAGON_API OpSchema {
 public:
  /*! \brief Default constructor */
  OpSchema() : op_type_("unknown"), file_("unknown"), line_(0) {
    Init();
  }

  /*! \brief Constructor with the defined spec */
  OpSchema(const string& op_type, const string& file, const int line)
      : op_type_(op_type), file_(file), line_(line) {
    Init();
  }

  /*! \brief Set a fixed number of inputs */
  OpSchema& NumInputs(int n);

  /*! \brief Set the min and max number of inputs */
  OpSchema& NumInputs(int min_num, int max_num);

  /*! \brief Set a fixed number of outputs */
  OpSchema& NumOutputs(int n);

  /*! \brief Set the min and max number of outputs */
  OpSchema& NumOutputs(int min_num, int max_num);

  /*! \brief Set the rule to allow inplace with a group of indices */
  OpSchema& AllowInplace(set<pair<int, int>> inplace);

  /*! \brief Set the rule to allow inplace with a function */
  OpSchema& AllowInplace(std::function<bool(int, int)> inplace);

  /*! \brief Check if the given def matches this schema */
  bool Verify(const OperatorDef& def) const;

  /*! \brief Check if the inplace is allowed */
  std::function<bool(int, int)> CheckInplace = [](int, int) { return false; };

 private:
  /*! \brief Initialize the default settings */
  void Init() {
    min_input_ = min_output_ = 0;
    max_input_ = max_output_ = std::numeric_limits<int>::max();
  }

  string op_type_, file_;
  int line_;
  int min_input_, max_input_;
  int min_output_, max_output_;
};

class DRAGON_API OpSchemaRegistry {
 public:
  /*! \brief Register an op schema */
  static OpSchema&
  NewSchema(const string& op_type, const string& file, const int line) {
    auto& m = schema_map();
    CHECK(!m.count(op_type))
        << "\nOpSchema(" << op_type << ") has registered before."
        << "\nat file: " << file << "\n   line: " << line;
    m.emplace(std::make_pair(op_type, OpSchema(op_type, file, line)));
    return m[op_type];
  }

  /*! \brief Return the specified op schema */
  static const OpSchema* Schema(const string& op_type) {
    auto& m = schema_map();
    if (m.count(op_type)) return &m[op_type];
    LOG(WARNING) << "OpSchema(" << op_type << ") has not registered yet.";
    return nullptr;
  }

 private:
  /*! \brief Return the global schema map */
  static Map<string, OpSchema>& schema_map();
};

#define OPERATOR_SCHEMA(name)                 \
  static OpSchema* ANONYMOUS_VARIABLE(name) = \
      &OpSchemaRegistry::NewSchema(#name, __FILE__, __LINE__)

} // namespace dragon

#endif // DRAGON_CORE_OPERATOR_SCHEMA_H_
