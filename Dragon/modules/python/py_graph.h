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

#ifndef DRAGON_PYTHON_PY_GRAPH_H_
#define DRAGON_PYTHON_PY_GRAPH_H_

#include "py_dragon.h"

namespace dragon {

namespace python {

void AddGraphMethods(pybind11::module& m) {
    /*! \brief Create a graph from the serialized def */
    m.def("CreateGraph", [](
        const string&           serialized,
        const bool              verbose) {
        GraphDef graph_def;
        if (!graph_def.ParseFromString(serialized))
            LOG(FATAL) << "Failed to parse the GraphDef.";
        auto* graph = ws()->CreateGraph(graph_def);
        if (verbose) {
            // It is not a good design to print the debug string
            auto* graph_tensor = ws()->CreateTensor(
                "/graph_def/optimized/" + graph->name());
            if (graph_tensor->count() > 0) {
                auto* data = graph_tensor->mutable_data<string, CPUContext>();
                std::cout << data[0] << std::endl;
            }

        }
        // Return the graph name may be different from the def
        // We will make a unique dummy name on creating the graph
        return graph->name();
    });

    /*! \brief Run an existing graph */
    m.def("RunGraph", [](
        const string&           name,
        const string&           include,
        const string&           exclude) {
        pybind11::gil_scoped_release g;
        ws()->RunGraph(name, include, exclude);
    });

    /*! \brief List all of the existing graphs */
    m.def("Graphs", []() { ws()->GetGraphs(); });
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_GRAPH_H_