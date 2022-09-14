#include "dragon/core/workspace.h"

namespace dragon {

Workspace* CPUContext::workspace() {
  static thread_local Workspace workspace("");
  return &workspace;
}

} // namespace dragon
