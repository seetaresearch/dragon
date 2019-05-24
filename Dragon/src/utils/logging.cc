#include <map>
#include <cstdlib>

#include "utils/string.h"
#include "utils/logging.h"

namespace dragon {

LogSeverity g_log_destination = INFO;

std::string LOG_SEVERITIES[] = {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "FATAL"
};

std::map<std::string, LogSeverity> LOG_LEVELS = {
    { "DEBUG", DEBUG },
    { "INFO", INFO },
    { "WARNING", WARNING },
    { "ERROR", ERROR },
    { "FATAL", FATAL }
};

std::map<std::string, int> g_log_count, g_log_every;

void SetLogDestination(LogSeverity type) {
    g_log_destination = type;
}

LogSeverity StrToLogSeverity(std::string level) {
    return LOG_LEVELS[level];
}

std::string GenLogHashKey(const char* file, int line) {
    return std::string(file) + str::to(line);
}

int EveryNRegister(
    const char*             file,
    int                     line,
    int                     severity,
    int                     n) {
    std::string hash_key = GenLogHashKey(file, line);
    if (!g_log_every.count(hash_key)) g_log_every[hash_key] = n;
    if (++g_log_count[hash_key] == g_log_every[hash_key]) {
        g_log_count[hash_key] = 0;
        return severity;
    } else {
        return -1;
    }
}

MessageLogger::MessageLogger(
    const char*             file,
    int                     line,
    int                     severity)
    : severity_(severity) {
    if (severity < g_log_destination) return;
    std::string filename_only;
    StripBasename(file, &filename_only);
    stream_ << LOG_SEVERITIES[severity] << " " 
            << filename_only << ":" << line << "] ";
}

MessageLogger::~MessageLogger() {
    if (severity_ < g_log_destination) return;
    stream_ << "\n";
    std::cerr << stream_.str() << std::flush;
    if (severity_ == FATAL) {
        std::cerr << "*** Check failure stack trace : ***" << std::endl;
        abort();
    }
}

void MessageLogger::StripBasename(
    const std::string&      full_path,
    std::string*            filename) {
    size_t pos1 = full_path.rfind('/');
    size_t pos2 = full_path.rfind('\\');
    size_t pos = std::string::npos;
    if (pos1 != std::string::npos) pos = pos1;
    if (pos2 != std::string::npos) pos = pos2;
    if (pos != std::string::npos) {
        *filename = full_path.substr(pos + 1, std::string::npos);
    } else  { *filename = full_path; }
}

}  // namespace dragon