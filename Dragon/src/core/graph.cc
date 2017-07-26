#include "core/operator_schema.h"
#include "core/graph.h"
#include "core/workspace.h"

namespace dragon {

GraphBase::GraphBase(const GraphDef& graph_def, Workspace* ws)
    : name_(graph_def.name()), ws_(ws) {
    for (auto arg : graph_def.arg()) {
        CHECK_GT(arg.name().size(), 0);
        CHECK_EQ(args_.count(arg.name()), 0);
        args_[arg.name()] = arg;
    }
    Set<string> known_tensors;

    //  topo-check for a graph
    for (const OperatorDef& op : graph_def.op()) {
        //  check inputs
        for (auto& in : op.input())
            CHECK(known_tensors.count(in) || ws_->HasTensor(in))
                << "input: " << in << " for op: " 
                << op.name() << " is unknown.";
        //  add outputs
        for (auto& out : op.output()) known_tensors.insert(out);
    }

    //  check for all objective targets
    Set<string> objective_targets;
    for (auto& target : graph_def.target()) {
        CHECK(known_tensors.count(target) || ws_->HasTensor(target))
            << "target: " << target << " does not exist in computional graph.";
        objective_targets.insert(target);
    }

    //  check for all gradient targets
    for (auto& g_target : graph_def.g_target()) {
        string cost = g_target.cost();
        string wrt = g_target.wrt();
        CHECK(known_tensors.count(cost) || ws_->HasTensor(cost))
            << "target: " << cost << "_grad does not exist in computional graph.";
        CHECK(known_tensors.count(wrt) || ws_->HasTensor(wrt))
            << "target: " << wrt << "_grad does not exist in computional graph.";
        CHECK_GT(objective_targets.count(cost), 0)
            << "\nto solve d(" << cost << ")/d(" << wrt << "), "
            << "you must set " << cost
            << "\nas a objective tensor to solve before derivating it.";
    }
}

void Graph::ForwardShareDyeing(string u, string ancestor) {
    if (renamed_.count(u)) return;
    renamed_[u] = ancestor;
    if (dag_[u].childs.size() == 1) {
        string op_type = dag_[dag_[u].childs[0]].op_type;
        auto* schema = OpSchemaRegistry::Schema(op_type);
        if (schema->AllowInplace())
            ForwardShareDyeing(dag_[u].childs[0], ancestor);
    }        
}

void Graph::ForwardPruneDyeing(string u, string leaf, vector<string> path) {
    if (visited_.count(u)) {
        if (visited_[u]) 
            for (auto& node : path) 
                visited_[node] = colored_[node] = true;
        return;
    }
    visited_[u] = false;
    for (int i = 0; i < dag_[u].childs.size(); i++) {
        string v = dag_[u].childs[i];
        vector<string> new_path(path);
        new_path.push_back(v);
        if (v == leaf) {
            for (auto& node : new_path) 
                visited_[node] = colored_[node] = true;
            return;
        }
        ForwardPruneDyeing(v, leaf, new_path);
    }
}

void Graph::BackwardPruneDyeing(string v) {
    colored_[v] = true;
    for (int i = 0; i < dag_[v].parents.size(); i++) {
        string u = dag_[v].parents[i];
        if (colored_.count(u)) continue;
        BackwardPruneDyeing(u);
    }
}

GraphDef Graph::Prune(const GraphDef& graph_def) {
    dag_.clear(); colored_.clear();
    //  build Graph
    for (int i = 0; i< graph_def.op_size(); i++) {
        const OperatorDef& op = graph_def.op(i);
        for (auto& v : op.output()) {
            vector<string> sp_u;
            if (!op.input_size()) sp_u.resize(op.output_size(), "");
            else sp_u.assign(op.input().begin(), op.input().end());
            for (auto& u : sp_u) {
                if (u == "ignore") continue;
                dag_[v].parents.push_back(u);
                dag_[u].childs.push_back(v);
                dag_[v].op_idx = i;
            }
            dag_[v].op_type = op.type();
        }
    }

    //  backward dyeing for all objective targets (e.g loss)
    for (int i = 0; i < graph_def.target_size(); i++) {
        targets_.insert(graph_def.target(i));
        if (colored_[graph_def.target(i)]) continue;
        BackwardPruneDyeing(graph_def.target(i));
    }

    //  forward dyeing through connected path for all gradient targets
    for (int i = 0; i < graph_def.g_target_size(); i++) {
        targets_.insert(graph_def.g_target(i).wrt() + "_grad");
        string u = graph_def.g_target(i).cost() + "_grad";
        string v = graph_def.g_target(i).wrt() + "_grad";
        if (!graph_def.g_target(i).external().empty())
            v = graph_def.g_target(i).external();
        if (ws()->HasTensor(u)) u = ws()->GetTensor(u)->name();
        if (ws()->HasTensor(v)) v = ws()->GetTensor(v)->name();
        visited_.clear();
        ForwardPruneDyeing(u, v, vector<string>({ u }));
    }

    //  select all dyed operators
    set<int> selected_op_indices;    //  note that we use set to keep topo-order
    for (auto it : colored_) {
        if (dag_[it.first].op_idx == -1) continue;
        selected_op_indices.insert(dag_[it.first].op_idx);
    }

    //  ensure that the update target will not be removed(colored_ it)
    for (int i = 0; i < graph_def.u_target_size(); i++) {
        UpdateTarget target = graph_def.u_target(i);
        for (auto& tensor : target.tensor())
            colored_[tensor] = true;
    }

    //  remove the tensors that can not be produced(redundant)
    Set<string> outputs;
    //  check if having feeded tensors
    for (auto& tensor : ws()->GetTensors()) outputs.insert(tensor);
    map<int, OperatorDef> ops_final;  //  note that we use map to keep topo-order

    for (auto it : selected_op_indices) {
        OperatorDef op_def;
        op_def.CopyFrom(graph_def.op(it));
        //  handle inputs
        for (int i = 0; i < graph_def.op(it).input_size(); i++){
            string input = graph_def.op(it).input(i);
            if (!colored_[input] || !outputs.count(input))
                *op_def.mutable_input(i) = "ignore";
        }
        //  handle outputs
        for (int i = 0; i < graph_def.op(it).output_size(); i++){
            string output = graph_def.op(it).output(i);
            if (!colored_[output]) *op_def.mutable_output(i) = "ignore";
            else outputs.insert(op_def.output(i));
        }
        ops_final[it].CopyFrom(op_def);
    }

    //  build the pruned graph
    GraphDef prune_graph;
    prune_graph.CopyFrom(graph_def);
    prune_graph.clear_op();
    for (auto it : ops_final) prune_graph.add_op()->CopyFrom(it.second);
    return prune_graph;
}

GraphDef Graph::Share(const GraphDef& graph_def) {
    renamed_.clear();

    //  forward dyeing to check all available sharing tensors
    for (int i = 0; i< graph_def.op_size(); i++) {
        const OperatorDef& op = graph_def.op(i);
        for (auto& u : op.input()) ForwardShareDyeing(u, u);
        for (auto& v : op.output()) ForwardShareDyeing(v, v);
    }

    GraphDef share_graph;
    share_graph.CopyFrom(graph_def);

    //  rename to create in-place
    for (int i = 0; i < graph_def.op_size(); i++) {
        const OperatorDef& op = graph_def.op(i);
        for (int j = 0; j < op.input_size(); j++) {
            if (renamed_.count(op.input(j))) {
                *share_graph.mutable_op(i)->mutable_input(j) = renamed_[op.input(j)];
                ws()->CreateRename(op.input(j), renamed_[op.input(j)]);
            }
        }
        for (int j = 0; j < op.output_size(); j++) {
            if (renamed_.count(op.output(j))) {
                *share_graph.mutable_op(i)->mutable_output(j) = renamed_[op.output(j)];
                ws()->CreateRename(op.output(j), renamed_[op.output(j)]);
            }
        }
    }
    return share_graph;
}

GraphDef Graph::MakeUpdate(const GraphDef& graph_def) {
    GraphDef update_graph; 
    update_graph.CopyFrom(graph_def);
    OperatorDef async_update; async_update.set_type("AsyncUpdate");
    for (int i = 0; i < graph_def.u_target_size(); i++) {
        UpdateTarget target = graph_def.u_target(i);
        vector<string> missing_tensors;
        //    missing check
        for (auto& tensor : target.tensor()) {
            if (!ws()->HasTensor(tensor)) {
                LOG(INFO) << "missing Tensor: " << tensor;
                missing_tensors.push_back(tensor);
            }
        }
        if (missing_tensors.size() == 0) {
            vector<Argument> args;
            for (auto& arg : target.arg()) args.push_back(arg);
            OperatorDef op_def = MakeOperatorDef(target.type(), target.name(),
                vector < string >({ target.tensor(1) }),
                vector < string >({ target.tensor(0) }));
            async_update.add_input(target.tensor(1));
            async_update.add_output(target.tensor(0));
            op_def.mutable_arg()->CopyFrom(target.arg());
            async_update.mutable_arg()->CopyFrom(target.arg());
            update_graph.add_op()->CopyFrom(op_def);
        } else {
            LOG(INFO) << "missing tensors. skip update Tensor("
                      << target.tensor(0) << ")";
        }
    }
    bool allow_async = false;
    int group_size = 1;
    for (auto& arg : async_update.arg()) {
        if (arg.name() == "mode")
            if (arg.s() == "Async" || arg.s() == "Async_No_Lock") allow_async = true;
        if (arg.name() == "group_size") group_size = arg.i();
    }
    if (allow_async) {
        for (int i = 0; i < group_size; i++) {
            OperatorDef server_op; server_op.CopyFrom(async_update);
            Argument arg_node_id; arg_node_id.set_name("node_id");
            arg_node_id.set_i(i); server_op.add_arg()->CopyFrom(arg_node_id);
            update_graph.add_op()->CopyFrom(server_op);
        }
    }
    update_graph.clear_u_target();
    return update_graph;
}

bool Graph::Create(const GraphDef& graph_def, Workspace* ws) {
    bool has_device_option = graph_def.has_device_option();
    bool has_debug_mode = graph_def.has_debug_mode();
    for (const OperatorDef& plain_op_def: graph_def.op()) {
        OperatorDef op_def(plain_op_def);
        LOG(DEBUG) << "Create Operator " << plain_op_def.name() 
                   << ": " << plain_op_def.type();

        //  inherit device option if necessary
        if (!op_def.has_device_option() && has_device_option) 
            op_def.mutable_device_option()->CopyFrom(graph_def.device_option());

        //  inherit debug mode if necessary
        if (!op_def.has_debug_mode() && has_debug_mode)
            op_def.set_debug_mode(graph_def.debug_mode());

        OperatorBase* op = CreateOperator(op_def, ws);
        ops_.push_back(op);
    }
    return true;
}

Graph::Graph(const GraphDef& graph_def, Workspace* ws)
    : GraphBase(graph_def, ws) {
    GraphDef optimized_graph;
    if (graph_def.u_target_size() > 0) {
        //  check if existing any update requests
        //  note that graph with update ops is not a dag
        //  we handle them independently
        optimized_graph = MakeUpdate(graph_def);
    } else {
        optimized_graph = Prune(graph_def);
        optimized_graph = Share(optimized_graph);
    }

    //  store the final graph as a tensor for python to draw
    Tensor* string_tensor = ws_->CreateTensor("GraphDef_" + optimized_graph.name());
    string_tensor->Reshape(vector<TIndex>(1, 1));
    string* data = string_tensor->mutable_data<string, CPUContext>();
    data[0] = optimized_graph.SerializeAsString();

    //  create
    Create(optimized_graph, ws);
}

bool Graph::Run(const string& include, const string& exclude) {
    for (auto op : ops_) {
        if (!include.empty()) 
            if (op->type().find(include) == string::npos) continue;
        if (!exclude.empty())
            if (op->type().find(exclude) != string::npos) continue;
        op->SwitchToPhase(this->args_["phase"].s());
        op->Run(); 
    }
    return true;
}

DEFINE_REGISTRY(GraphRegistry, GraphBase, const GraphDef&, Workspace*);

GraphBase* NewGraph(const GraphDef& graph_def, Workspace* ws) {
    if (!graph_def.has_graph_type()) return new Graph(graph_def, ws);
    return GraphRegistry()->Create(graph_def.graph_type(), graph_def, ws);
}

}    // namespace dragon