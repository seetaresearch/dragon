#include "py_graph.h"
#include "py_autograd.h"
#include "py_operator.h"
#include "py_tensor.h"
#include "py_cuda.h"
#include "py_mpi.h"
#include "py_io.h"
#include "py_onnx.h"
#include "py_config.h"
#include "py_proto.h"

namespace dragon {

namespace python {

DEFINE_TYPED_REGISTRY(TensorFetcherRegistry, TypeId, TensorFetcherBase);
DEFINE_TYPED_REGISTRY(TensorFeederRegistry, TypeId, TensorFeederBase);

Map<string, unique_ptr < Workspace > > g_workspaces;
Map<string, vector<string> > sub_workspaces;
Workspace* g_workspace;
string g_current_workspace;

Workspace* ws() { return g_workspace; }

TypeId CTypeToFetcher(TypeId type) {
    static Map<TypeId,TypeId> c_type_map {
        { TypeMeta::Id<bool>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<int8_t>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<uint8_t>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<int>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<int64_t>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<float16>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<float>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<double>(), TypeMeta::Id<NumpyFetcher>() },
        { TypeMeta::Id<string>(), TypeMeta::Id<StringFetcher>() }};
    return c_type_map.count(type) ? c_type_map[type] : 0;
}

REGISTER_TENSOR_FETCHER(TypeMeta::Id<NumpyFetcher>(), NumpyFetcher);
REGISTER_TENSOR_FETCHER(TypeMeta::Id<StringFetcher>(), StringFetcher);
REGISTER_TENSOR_FEEDER(TypeMeta::Id<NumpyFeeder>(), NumpyFeeder);

void SwitchWorkspace(
    const string&           name,
    const bool              create_if_missing = true) {
    if (g_workspaces.count(name)) {
        g_current_workspace = name;
        g_workspace = g_workspaces[name].get();
    } else if (create_if_missing) {
        unique_ptr<Workspace> new_workspace(new Workspace(name));
        g_workspace = new_workspace.get();
        g_workspaces[name] = std::move(new_workspace);
        sub_workspaces[name] = vector<string>();
        g_current_workspace = name;
    } else {
        LOG(FATAL) << "Workspace of the given name does not exist."
           "\nAnd, it is not allowed to create. (Try to alllow?)";
    }
}

void OnImportModule() {
    []() { import_array1(); }();
    static bool initialized = false;
    if (initialized) return;
    SwitchWorkspace("default", true);
    g_current_workspace = "default";
    initialized = true;
}

PYBIND11_MODULE(libdragon, m) {

    /* ------------------------------------ *
     *                                      *
     *              Workspace               *
     *                                      *
     * ------------------------------------ */

     /*! \brief Switch to the specific workspace */
    m.def("SwitchWorkspace", &SwitchWorkspace);

    /*! \brief Return the current active workspace */
    m.def("CurrentWorkspace", []() {
        return g_current_workspace;
    });

    /*! \brief List all of the existing workspace */
    m.def("Workspaces", []() -> vector<string> {
        vector<string> names;
        for (auto const& it : g_workspaces)
            names.emplace_back(it.first);
        return names;
    });

    /*! \brief Move the source workspace into the target */
    m.def("MoveWorkspace", [](
        const string&           target,
        const string&           source) {
        CHECK(g_workspaces.count(source))
            << "\nSource Workspace(" << source << ") does not exist.";
        CHECK(g_workspaces.count(target))
            << "\nTarget Workspace(" << target << ") does not exist.";
        g_workspaces[target]->Move(g_workspaces[source].get());
        sub_workspaces[target].push_back(source);
        LOG(INFO) << "Move the Workspace(" << source << ") "
            << "into the Workspace(" << target << ").";
    });

    /*! \brief Reset the specific workspace */
    m.def("ResetWorkspace", [](const string& name) {
        string target_workspace = g_current_workspace;
        if (!name.empty()) target_workspace = name;
        CHECK(g_workspaces.count(target_workspace))
            << "\nWorkspace(" << target_workspace
            << ") does not exist, can not be reset.";
        LOG(INFO) << "Reset the Workspace(" << target_workspace << ")";
        g_workspaces[target_workspace].reset(new Workspace(target_workspace));
        g_workspace = g_workspaces[target_workspace].get();
        for (auto& sub_workspace : sub_workspaces[target_workspace]) {
            if (g_workspaces.count(sub_workspace) > 0)
                g_workspace->Move(g_workspaces[sub_workspace].get());
        }
    });

    /*! \brief Release the memory of tensors */
    m.def("ClearWorkspace", [](const string& name) {
        string target_workspace = g_current_workspace;
        if (!name.empty()) target_workspace = name;
        CHECK(g_workspaces.count(target_workspace))
            << "\nWorkspace(" << target_workspace
            << ") does not exist, can not be reset.";
        LOG(INFO) << "Clear the Workspace(" << target_workspace << ")";
        g_workspaces[target_workspace]->Clear();
    });

    m.def("FeedTensor", [](
        const string&           name,
        pybind11::object        value,
        const string&           device_option) {
        DeviceOption dev;
        if (!device_option.empty()) {
            if (!dev.ParseFromString(device_option)) {
                LOG(FATAL) << "Failed to parse the DeviceOption.";
            }
        }
        Tensor* tensor = g_workspace->CreateTensor(name);
        unique_ptr<TensorFeederBase> feeder(TensorFeederRegistry()
            ->Create(TypeMeta::Id<NumpyFeeder>()));
        feeder->Feed(dev, reinterpret_cast<
            PyArrayObject*>(value.ptr()), tensor);
    });

    m.def("FetchTensor", [](const string& name) {
        if (!g_workspace->HasTensor(name))
            LOG(FATAL) << "Tensor(" + name + ") "
                "does not exist. Have you registered it?";
        Tensor* tensor = g_workspace->GetTensor(name);
        TypeId type_id = CTypeToFetcher(tensor->meta().id());
        CHECK(type_id != 0)
            << "\nTensor(" << tensor->name()
            << ") does not initialize or had been reset.";
        unique_ptr<TensorFetcherBase> fetcher(CreateFetcher(type_id));
        if (fetcher.get()) {
            // Copy the tensor data to a numpy object
            return fetcher->Fetch(*tensor);
        } else {
            LOG(FATAL) << name << " is not a C++ native type.";
            return pybind11::object();
        }
    });

    /*!            Misc             */
    m.def("GetDummyName", [](
        const string&           basename,
        const string&           suffix,
        const string&           domain,
        const bool              zero_based) {
        return ws()->GetDummyName(
            basename, suffix, domain, zero_based);
    });

    AddIOMethods(m);
    AddMPIMethods(m);
    AddCUDAMethods(m);
    AddProtoMethods(m);
    AddGraphMethods(m);
    AddTensorMethods(m);
    AddConfigMethods(m);
    AddGradientMethods(m);
    AddOperatorMethods(m);

    OnImportModule();
    m.def("OnModuleExit", []() { g_workspaces.clear(); });
}

}  // namespace python

}  // namespace dragon