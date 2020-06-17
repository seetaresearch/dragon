#include <cstdlib>
#include <unordered_map>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

#include "dragon/utils/logging.h"
#include "dragon/utils/string_utils.h"

namespace dragon {

namespace {

LogSeverity destination = LogSeverity::INFO;
std::unordered_map<std::string, int> log_count, log_every;

std::string GetServerity(int severity) {
  static std::string severities[] = {
      "DEBUG", "INFO", "WARNING", "ERROR", "FATAL"};
  return severities[severity];
}

LogSeverity GetServerity(const std::string& severity) {
  static std::unordered_map<std::string, LogSeverity> severities = {
      {"DEBUG", DEBUG},
      {"INFO", INFO},
      {"WARNING", WARNING},
      {"ERROR", ERROR},
      {"FATAL", FATAL}};
  return severities[severity];
}

void StripBasename(const std::string& path, std::string* filename) {
  size_t pos1 = path.rfind('/');
  size_t pos2 = path.rfind('\\');
  size_t pos = std::string::npos;
  if (pos1 != std::string::npos) pos = pos1;
  if (pos2 != std::string::npos) pos = pos2;
  if (pos != std::string::npos) {
    *filename = path.substr(pos + 1, std::string::npos);
  } else {
    *filename = path;
  }
}

} // namespace

void SetLogDestination(LogSeverity severity) {
  destination = severity;
}

void SetLogDestination(const std::string& severity) {
  destination = GetServerity(severity);
}

int LogEveryRegister(const char* file, int line, int severity, int n) {
  auto hash_key = std::string(file) + str::to(line);
  if (!log_every.count(hash_key)) log_every[hash_key] = n;
  if (++log_count[hash_key] == log_every[hash_key]) {
    log_count[hash_key] = 0;
    return severity;
  } else {
    return -1;
  }
}

MessageLogger::MessageLogger(const char* file, int line, int severity)
    : severity_(severity) {
  if (severity < destination) return;
  if (line < 0) return; // Remove
  std::string filename;
  StripBasename(file, &filename);
  stream_ << GetServerity(severity) << " " << filename << ":" << line << "] ";
}

MessageLogger::~MessageLogger() {
  if (severity_ < destination) return;
  stream_ << "\n";
#if defined(__ANDROID__)
  static const int android_log_levels[] = {
      ANDROID_LOG_DEBUG,
      ANDROID_LOG_INFO,
      ANDROID_LOG_WARN,
      ANDROID_LOG_ERROR,
      ANDROID_LOG_FATAL,
  };
  __android_log_write(
      android_log_levels[severity_], "libdragonrt", stream_.str().c_str());
  if (severity_ == LogSeverity::FATAL) {
    __android_log_write(
        ANDROID_LOG_FATAL,
        "libdragonrt",
        "*** Check failure stack trace : ***\n");
    abort();
  }
#else
  std::cerr << stream_.str() << std::flush;
  if (severity_ == LogSeverity::FATAL) {
    std::cerr << "*** Check failure stack trace : ***" << std::endl;
    abort();
  }
#endif
}

} // namespace dragon
