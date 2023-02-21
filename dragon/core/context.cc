#include "dragon/core/workspace.h"

namespace dragon {

Workspace* CPUContext::workspace() {
  static thread_local Workspace workspace("");
  return &workspace;
}

std::mutex& CPUContext::mutex() {
  static std::mutex m;
  return m;
}

CPUObjects& CPUContext::objects() {
  static thread_local CPUObjects objects_;
  return objects_;
}

} // namespace dragon
