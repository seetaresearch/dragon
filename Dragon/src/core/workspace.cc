#include "core/operator.h"
#include "core/graph.h"
#include "core/workspace.h"

namespace dragon {

GraphBase* Workspace::CreateGraph(const GraphDef& graph_def) {
    CHECK(graph_def.has_name());
    if (graph_map_.count(graph_def.name()))
        return graph_map_[graph_def.name()].get();
    LOG(DEBUG) << "Create Graph: " << graph_def.name();
    graph_map_[graph_def.name()] = unique_ptr<GraphBase>(NewGraph(graph_def, this));
    return graph_map_[graph_def.name()].get();
}

Workspace::~Workspace() {
    for (int i = 0; i < WORKSPACE_MAX_CORRUPTED_SIZE; i++) {
        string name = "_t_mirrow_stage_buffer_" + dragon_cast<string, int>(i);
        if (HasTensor(name)) {
            MixedMemory* mem = GetTensor(name)->memory();
            if (mem != nullptr) delete mem;
        }
    }
}

}    // namespace dragon