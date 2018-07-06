#include "py_graph.h"
#include "py_autograd.h"
#include "py_operator.h"
#include "py_tensor.h"
#include "py_cuda.h"
#include "py_mpi.h"
#include "py_io.h"
#include "py_config.h"

DEFINE_TYPED_REGISTRY(TensorFetcherRegistry, TypeId, TensorFetcherBase);
DEFINE_TYPED_REGISTRY(TensorFeederRegistry, TypeId, TensorFeederBase);

Map<string, unique_ptr < Workspace > > g_workspaces;
Map<string, vector<string> > sub_workspaces;
Workspace* g_workspace;
string g_current_workspace;

Workspace* ws() { return g_workspace; }

TypeId CTypeToFetcher(TypeId type) {
    static Map<TypeId,TypeId> c_type_map {
            { TypeMeta::Id<uint8_t>(), TypeMeta::Id<NumpyFetcher>() },
            { TypeMeta::Id<int>(), TypeMeta::Id<NumpyFetcher>() },
            { TypeMeta::Id<int64_t>(), TypeMeta::Id<NumpyFetcher>() },
            { TypeMeta::Id<float>(), TypeMeta::Id<NumpyFetcher>() },
            { TypeMeta::Id<double>(), TypeMeta::Id<NumpyFetcher>() },
            { TypeMeta::Id<float16>(), TypeMeta::Id<NumpyFetcher>() },
            { TypeMeta::Id<string>(), TypeMeta::Id<StringFetcher>() }};
    return c_type_map.count(type) ? c_type_map[type] : 0;
}

REGISTER_TENSOR_FETCHER(TypeMeta::Id<NumpyFetcher>(), NumpyFetcher);
REGISTER_TENSOR_FETCHER(TypeMeta::Id<StringFetcher>(), StringFetcher);
REGISTER_TENSOR_FEEDER(TypeMeta::Id<NumpyFeeder>(), NumpyFeeder);

extern "C" {

bool SwitchWorkspaceInternal(
    const string& name,
    const bool create_if_missing) {
    if (g_workspaces.count(name)) {
        g_current_workspace = name;
        g_workspace = g_workspaces[name].get();
        return true;
    } else if (create_if_missing) {
        unique_ptr<Workspace> new_workspace(new Workspace(name));
        g_workspace = new_workspace.get();
        g_workspaces[name] = std::move(new_workspace);
        sub_workspaces[name] = vector<string>();
        g_current_workspace = name;
        return true;
    } else {
        return false;
    }
}

inline PyObject* SwitchWorkspaceCC(PyObject* self, PyObject *args) {
    char* cname;
    PyObject* create_if_missing = nullptr;
    if (!PyArg_ParseTuple(args, "s|O", &cname, &create_if_missing)) {
        PyErr_SetString(PyExc_ValueError, 
            "Excepted a workspace name and a optional "
            "boolean value that specific whether to create if missing.");
        return nullptr;
    }
    bool success = SwitchWorkspaceInternal(cname,
        PyObject_IsTrue(create_if_missing) ? true : false);
    if (!success) {
        PyErr_SetString(PyExc_RuntimeError, 
            "Workspace of the given name does not exist."
            "\nAnd, it is not allowed to create. (Try alllow?)");
        return nullptr;
    }
    Py_RETURN_TRUE;
}

inline PyObject* MoveWorkspaceCC(PyObject* self, PyObject *args) {
    char* target_ws, *src_ws;
    if (!PyArg_ParseTuple(args, "ss", &target_ws, &src_ws)) {
        PyErr_SetString(PyExc_ValueError, 
            "Excepted the target and source workspace.");
        return nullptr;
    }
    CHECK(g_workspaces.count(src_ws))
        << "\nSource Workspace(" << src_ws << ") does not exist.";
    CHECK(g_workspaces.count(target_ws))
        << "\nTarget Workspace(" << target_ws << ") does not exist.";
    g_workspaces[target_ws]->MoveWorkspace(g_workspaces[src_ws].get());
    sub_workspaces[target_ws].push_back(string(src_ws));
    LOG(INFO) << "Move the Workspace(" << src_ws << ") into the "
              << "Workspace(" << target_ws << ").";
    Py_RETURN_TRUE;
}

inline PyObject* CurrentWorkspaceCC(PyObject* self, PyObject* args) {
    return String_AsPyUnicode(g_current_workspace);
}

inline PyObject* WorkspacesCC(PyObject* self, PyObject* args) {
    PyObject* list = PyList_New(g_workspaces.size());
    int i = 0;
    for (auto const& it : g_workspaces) 
        CHECK_EQ(PyList_SetItem(list, i++, String_AsPyUnicode(it.first)), 0);
    return list;
}

inline PyObject* ResetWorkspaceCC(PyObject* self, PyObject* args) {
    char* cname;
    if (!PyArg_ParseTuple(args, "s", &cname)) {
        PyErr_SetString(PyExc_ValueError, 
            "Excepted a name to locate the workspace.");
        return nullptr;
    }
    string target_workspace = g_current_workspace;
    if (!string(cname).empty()) target_workspace = string(cname);
    CHECK(g_workspaces.count(target_workspace))
        << "\nWorkspace(" << target_workspace
        << ") does not exist, can not be reset.";
    LOG(INFO) << "Reset the Workspace(" << target_workspace << ")";
    g_workspaces[target_workspace].reset(new Workspace(target_workspace));
    g_workspace = g_workspaces[target_workspace].get();
    for (auto& sub_workspace : sub_workspaces[target_workspace]) {
        if (g_workspaces.count(sub_workspace) > 0)
            g_workspace->MoveWorkspace(g_workspaces[sub_workspace].get());
    }
    Py_RETURN_TRUE;
}

inline PyObject* ClearWorkspaceCC(PyObject* self, PyObject* args) {
    char* cname;
    if (!PyArg_ParseTuple(args, "s", &cname)) {
        PyErr_SetString(PyExc_ValueError, 
            "Excepted a name to locate the workspace.");
        return nullptr;
    }
    string target_workspace = g_current_workspace;
    if (!string(cname).empty()) target_workspace = string(cname);
    CHECK(g_workspaces.count(target_workspace))
        << "\nWorkspace(" << target_workspace
        << ") does not exist, can not be reset.";
    LOG(INFO) << "Clear the Workspace(" << target_workspace << ")";
    g_workspaces[target_workspace]->ClearWorkspace();
    Py_RETURN_TRUE;
}

inline PyObject* FetchTensorCC(PyObject* self, PyObject* args) {
    char* cname;
    if (!PyArg_ParseTuple(args, "s", &cname)) {
        PyErr_SetString(PyExc_ValueError, "Excepted a tensor name.");
        return nullptr;
    }
    if (!g_workspace->HasTensor(string(cname))) {
        PyErr_SetString(PyExc_ValueError, 
            "Tensor does not exist. Have you registered it?");
        return nullptr;
    }
    Tensor* tensor = g_workspace->GetTensor(string(cname));
    TypeId type_id = CTypeToFetcher(tensor->meta().id());
    CHECK(type_id != 0)
        << "\nTensor(" << tensor->name()
        << ") does not initialize or had been reset.";
    unique_ptr<TensorFetcherBase> fetcher(CreateFetcher(type_id));
    if (fetcher.get()) {
        // copy the tensor data to a numpy object
        return fetcher->Fetch(*tensor);
    } else {
        LOG(INFO) << string(cname) << " is not a C++ native type.";
        return nullptr;
    }
}

inline PyObject* FeedTensorCC(PyObject* self, PyObject* args) {
    char* cname;
    PyArrayObject* array = nullptr;
    PyObject *device_option = nullptr;
    if (!PyArg_ParseTuple(args, "sO|O", &cname, &array, &device_option)) {
        PyErr_SetString(PyExc_ValueError, 
            "Excepted the name, values and serialized DeviceOption.");
        return nullptr;
    }
    DeviceOption option;
    if (device_option != nullptr) {
        if (!option.ParseFromString(PyBytes_AsStringEx(device_option))) {
            PyErr_SetString(PyExc_ValueError, 
                "Failed to parse the DeviceOption.");
            return nullptr;
        }
    }
    Tensor* tensor = g_workspace->CreateTensor(string(cname));
    unique_ptr<TensorFeederBase> feeder(TensorFeederRegistry()
        ->Create(TypeMeta::Id<NumpyFeeder>()));
    if (feeder.get()) {
        return feeder->Feed(option, array, tensor);
    } else {
        PyErr_SetString(PyExc_TypeError, "Unknown device type.");
        return nullptr;
    }
}

#define PYFUNC(name) {#name, name, METH_VARARGS, ""}
#define PYENDFUNC {nullptr, nullptr, 0, nullptr}

PyMethodDef* GetAllMethods() {
    static PyMethodDef g_python_methods[] {
        /****  Workspace  ****/
        PYFUNC(SwitchWorkspaceCC),
        PYFUNC(MoveWorkspaceCC),
        PYFUNC(CurrentWorkspaceCC),
        PYFUNC(WorkspacesCC),
        PYFUNC(ResetWorkspaceCC),
        PYFUNC(ClearWorkspaceCC),
        /****  Graph  ****/
        PYFUNC(CreateGraphCC),
        PYFUNC(RunGraphCC),
        PYFUNC(GraphsCC),
        /****  AutoGrad  ****/
        PYFUNC(CreateGradientDefsCC),
        PYFUNC(RunGradientFlowCC),
        /****  Operator  ****/
        PYFUNC(RegisteredOperatorsCC),
        PYFUNC(NoGradientOperatorsCC),
        PYFUNC(RunOperatorCC),
        PYFUNC(RunOperatorsCC),
        PYFUNC(CreatePersistentOpCC),
        PYFUNC(RunPersistentOpCC),
        /****  Tensor  ****/
        PYFUNC(HasTensorCC),
        PYFUNC(CreateTensorCC),
        PYFUNC(CreateFillerCC),
        PYFUNC(GetFillerTypeCC),
        PYFUNC(RenameTensorCC),
        PYFUNC(TensorFromShapeCC),
        PYFUNC(TensorFromPyArrayCC),
        PYFUNC(GetTensorNameCC),
        PYFUNC(GetTensorInfoCC),
        PYFUNC(FeedTensorCC),
        PYFUNC(FetchTensorCC),
        PYFUNC(ToCPUTensorCC),
        PYFUNC(ToCUDATensorCC),
        PYFUNC(TensorToPyArrayCC),
        PYFUNC(TensorToPyArrayExCC),
        PYFUNC(ResetTensorCC),
        PYFUNC(TensorsCC),
        /****  MPI  ****/
        PYFUNC(MPIInitCC),
        PYFUNC(MPIRankCC),
        PYFUNC(MPISizeCC),
        PYFUNC(MPICreateGroupCC),
        PYFUNC(MPIFinalizeCC),
        /****  CUDA  ****/
        PYFUNC(IsCUDADriverSufficientCC),
        /****  I/O  ****/
        PYFUNC(RestoreCC),
        PYFUNC(SnapshotCC),
        /****  Config ****/
        PYFUNC(SetLogLevelCC),
        PYENDFUNC,
    };
    return g_python_methods;
}

static void import_array_wrapper() { import_array1(); }

void common_init() {
    import_array_wrapper();
    static bool initialized = false;
    if (initialized) return;
    SwitchWorkspaceInternal("default", true);
    g_current_workspace = "default";
    initialized = true;
}

#ifdef WITH_PYTHON3
static struct PyModuleDef libdragon = { PyModuleDef_HEAD_INIT,
                                        "libdragon", "", -1,
                                        GetAllMethods() };

PyMODINIT_FUNC PyInit_libdragon(void) {
    PyObject* module = PyModule_Create(&libdragon);
    if (module == nullptr) return nullptr;
    common_init();
    return module;
}

#else   // WITH_PYTHON2
PyMODINIT_FUNC initlibdragon(void) {
    PyObject* moudle = Py_InitModule("libdragon", GetAllMethods());
    if (moudle == nullptr) return;
    common_init();
}
#endif

}