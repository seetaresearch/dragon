#include "core/operator.h"
#include "core/graph.h"
#include "core/workspace.h"

namespace dragon {

GraphBase* Workspace::CreateGraph(const GraphDef& meta_graph) {
    CHECK(meta_graph.has_name())
        << "The name of given meta graph should not be empty.";
    if (graph_map_.count(meta_graph.name()))
        return graph_map_[meta_graph.name()].get();
    LOG(DEBUG) << "Create Graph: " << meta_graph.name();
    graph_map_[meta_graph.name()] = unique_ptr<GraphBase>(NewGraph(meta_graph, this));
    return graph_map_[meta_graph.name()].get();
}

Workspace::~Workspace() {
    for (int i = 0; i < WORKSPACE_MAX_CORRUPTED_SIZE; i++) {
        string name = "/opt/mirror_stage/buffer_"
            + std::to_string(i);
        if (tensor_map_.count(name) > 0) {
            MixedMemory* mem = tensor_map_[name]->memory();
            if (mem != nullptr) delete mem;
        }
    }
}

}    // namespace dragon