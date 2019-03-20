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

#ifndef DRAGON_CORE_OPERATOR_SCHEMA_H_
#define DRAGON_CORE_OPERATOR_SCHEMA_H_

#include <limits>
#include <functional>

#include "common.h"

namespace dragon {

class OpSchema {
 public:
    OpSchema()
        : op_type_("unknown"), file_("unknown"), line_(0) {
        Init();
    }

    OpSchema(
        const string&               op_type,
        const string&               file, 
        const int                   line)
        : op_type_(op_type), file_(file), line_(line) {
        Init();
    }

    bool Verify(const OperatorDef& def) const;

    OpSchema& IgnoreVerify() {
        ignore_verify_ = true; 
        return *this; 
    }

    OpSchema& Inplace(set<pair<int, int> > inplace);
    std::function<bool(int, int)> CheckInplace;
    bool AllowInplace() const { return allow_inplace_; }

    OpSchema& NumInputs(int n);
    OpSchema& NumInputs(int min_num, int max_num);

    OpSchema& NumOutputs(int n);
    OpSchema& NumOutputs(int min_num, int max_num);

 private:
    void Init() {
        min_input_ = min_output_= 0;
        max_input_ = max_output_ = std::numeric_limits<int>::max();
        CheckInplace = [](int, int) { return false; };
        ignore_verify_ =  allow_inplace_ = false;
    }

    string op_type_, file_;
    int line_, min_input_, max_input_;
    int min_output_, max_output_;
    bool allow_inplace_, ignore_verify_;
};

class OpSchemaRegistry {
 public:
    static OpSchema& NewSchema(
        const string&               op_type,
        const string&               file,
        const int                   line) {
        auto& m = schema_map();
        CHECK(!m.count(op_type))
            << "\nOpSchema(" << op_type
            << ") has registered before."
            << "\nat file: " << file
            << "\n   line: " << line;
        m.emplace(std::make_pair(op_type,
            OpSchema(op_type, file, line)));
        return m[op_type];
    }

    static const OpSchema* Schema(const string& op_type) {
        auto& m = schema_map();
        if (m.count(op_type)) return &m[op_type];
        LOG(WARNING) << "OpSchema(" << op_type
                     << ") has not registered yet.";
        return nullptr;
    }

 private:
    static Map<string, OpSchema>& schema_map() {
        static Map<string, OpSchema> schema_map_;
        return schema_map_;
    }
};

#define OPERATOR_SCHEMA(name) \
    static OpSchema& ANONYMOUS_VARIABLE(name) = \
        OpSchemaRegistry::NewSchema(#name, __FILE__, __LINE__)

}  // namespace dragon

#endif  // DRAGON_CORE_OPERATOR_SCHEMA_H_