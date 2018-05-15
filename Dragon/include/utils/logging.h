// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_UTILS_LOGGING_H_
#define DRAGON_UTILS_LOGGING_H_

#include <iostream>
#include <sstream>

namespace dragon {

enum LogSeverity { DEBUG, INFO, WARNING, ERROR, FATAL };

LogSeverity StrToLogSeverity(std::string level);

std::string SeverityToStr(LogSeverity severity);

void SetLogDestination(LogSeverity type);

int EveryNRegister(const char* file, int line, int severity, int n);

class MessageLogger {
 public:
    MessageLogger(const char* file, int line, int severity);
    ~MessageLogger();

    std::stringstream& stream() { return stream_; }

 private:
    int severity_;
    std::stringstream stream_;
    void StripBasename(const std::string &full_path, std::string* filename);
};

#define FATAL_IF(condition) condition ? MessageLogger("-1", -1, -1).stream() : MessageLogger(__FILE__, __LINE__, FATAL).stream()
#define CHECK(condition) FATAL_IF(condition) << "Check failed: "#condition" "
#define CHECK_OP(val1,val2,op) \
    FATAL_IF(val1 op val2) << "Check failed: " #val1 " " #op " " #val2 " " << "(" << val1 <<" vs "<<val2 <<")"
#define CHECK_EQ(val1, val2) CHECK_OP(val1, val2, ==)
#define CHECK_NE(val1, val2) CHECK_OP(val1, val2, !=)
#define CHECK_GT(val1, val2) CHECK_OP(val1, val2, >)
#define CHECK_GE(val1, val2) CHECK_OP(val1, val2, >=)
#define CHECK_LT(val1, val2) CHECK_OP(val1, val2, <)
#define CHECK_LE(val1, val2) CHECK_OP(val1, val2, <=)
#define LOG(severity) MessageLogger(__FILE__, __LINE__, severity).stream()
#define LOG_IF(severity, condition) if(condition) MessageLogger(__FILE__, __LINE__, severity).stream()
#define LOG_EVERY_N(severity, n) MessageLogger(__FILE__, __LINE__, EveryNRegister(__FILE__, __LINE__, severity, n)).stream()

}    // namespace dragon

#endif // DRAGON_UTILS_LOGGING_H_