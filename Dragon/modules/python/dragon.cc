#include "dragon.h"
#include "py_mpi.h"

#include "utils/caffemodel.h"
#include "utils/logging.h"

DEFINE_TYPED_REGISTRY(TensorFetcherRegistry, TypeId, TensorFetcherBase);
DEFINE_TYPED_REGISTRY(TensorFeederRegistry, TypeId, TensorFeederBase);

Map<string, unique_ptr < Workspace > > g_workspaces;
Workspace* g_workspace;
string g_current_workspace;

int DragonToNumpyType(const TypeMeta& meta) {
    static Map<TypeId, int> numpy_type_map{
            { TypeMeta::Id<float>(), NPY_FLOAT32 },
            { TypeMeta::Id<int>(), NPY_INT32 },
            { TypeMeta::Id<int64_t>(), NPY_INT64 },
            { TypeMeta::Id<double>(), NPY_FLOAT64 },
            { TypeMeta::Id<float16>(), NPY_FLOAT16 },
            { TypeMeta::Id<uint8_t>(), NPY_UINT8  }};
    return numpy_type_map.count(meta.id()) ? numpy_type_map[meta.id()] : -1;
}

const TypeMeta& NumpyTypeToDragon(int numpy_type) {
    static Map<int, TypeMeta> dragon_type_map{
            { NPY_FLOAT32, TypeMeta::Make<float>() },
            { NPY_INT32, TypeMeta::Make<int>() },
            { NPY_INT64, TypeMeta::Make<int64_t>() },
            { NPY_FLOAT64, TypeMeta::Make<double>() },
            { NPY_FLOAT16, TypeMeta::Make<float16>() },
            { NPY_UINT8, TypeMeta::Make<uint8_t>() }};

    static TypeMeta unknown_type;
    return dragon_type_map.count(numpy_type) ? dragon_type_map[numpy_type] : unknown_type;
}

TypeId CTypeToFetcher(TypeId type) {
    static Map<TypeId,TypeId> c_type_map {
            { TypeMeta::Id<uint8_t>(), TypeMeta::Id<NumpyFetcher>() },
            { TypeMeta::Id<int>(), TypeMeta::Id<NumpyFetcher>() },
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

PyObject* RegisteredOperatorsCC(PyObject* self, PyObject* args) {
    set<string> all_keys;
    for (const auto& name : CPUOperatorRegistry()->keys()) all_keys.insert(name);
    PyObject* list = PyList_New(all_keys.size());
    int idx = 0;
    for (const string& name : all_keys) 
        CHECK_EQ(PyList_SetItem(list, idx++, StdStringToPyUnicode(name)), 0);
    return list;
}

PyObject* NoGradientOperatorsCC(PyObject* self, PyObject* args) {
    set<string> all_keys;
    for (const auto& name : NoGradientRegistry()->keys()) all_keys.insert(name);
    PyObject* list = PyList_New(all_keys.size());
    int idx = 0;
    for (const string& name : all_keys) 
        CHECK_EQ(PyList_SetItem(list, idx++, StdStringToPyUnicode(name)), 0);
    return list;
}

PyObject* CreateGradientDefsCC(PyObject* self, PyObject* args) {
    PyObject* def_string = nullptr;
    PyObject* g_outputs_py = nullptr;
    if (!PyArg_ParseTuple(args, "SO!", &def_string, &PyList_Type, &g_outputs_py)) {
        PyErr_SetString(PyExc_ValueError, "CreateGradientDefsCC requires an input that is a serialized "
                                          "OperatorDef protobuf, and a list containing the gradient of the op's output.");
         return nullptr;
    }
    OperatorDef def;
    if (!def.ParseFromString(PyBytesToStdString(def_string))) {
        PyErr_SetString(PyExc_ValueError, "provied string is not a valid Operator protobuf.");
        return nullptr;
    }
    if (!GradientRegistry()->Has(def.type())) {
        PyErr_SetString(PyExc_KeyError, "gradient is not registered before.");
        return nullptr;
    }

    vector<string> g_outputs;
    int size = PyList_Size(g_outputs_py);
    for (int i = 0; i < size; i++) {
        PyObject* obj = PyList_GetItem(g_outputs_py, i);
        if (obj == Py_None) g_outputs.push_back("ignore");
        else g_outputs.push_back(PyString_AsString(PyObject_Str(obj)));
    }

    Gradient grad = MakeGradientForOp(def, g_outputs);

    PyObject* g_ops_py = PyList_New(grad.ops.size());
    for (int i = 0; i < grad.ops.size(); i++) 
        CHECK_EQ(PyList_SetItem(g_ops_py, i, StdStringToPyBytes(grad.ops[i].SerializeAsString())), 0);

    PyObject* g_input_py = PyList_New(grad.g_inputs.size());
    for (int i = 0; i < grad.g_inputs.size(); i++) 
        CHECK_EQ(PyList_SetItem(g_input_py, i, StdStringToPyUnicode(grad.g_inputs[i])), 0);

    PyObject* defaults_py = PyList_New(grad.defaults.size());
    for (int i = 0; i < grad.defaults.size(); i++) 
        CHECK_EQ(PyList_SetItem(defaults_py, i, PyFloat_FromDouble(grad.defaults[i])), 0);

    return PyTuple_Pack(3, g_ops_py, g_input_py, defaults_py);
}

bool SwitchWorkspaceInternal(const string& name, const bool create_if_missing) {
    if (g_workspaces.count(name)) {
        g_current_workspace = name;
        g_workspace = g_workspaces[name].get();
        return true;
    } else if (create_if_missing) {
        unique_ptr<Workspace> new_workspace(new Workspace());
        g_workspace = new_workspace.get();
        g_workspaces[name] = std::move(new_workspace);
        g_current_workspace = name;
        return true;
    } else {
        return false;
    }
}

PyObject* SwitchWorkspaceCC(PyObject* self, PyObject *args) {
    char* cname;
    PyObject* create_if_missing = nullptr;
    if (!PyArg_ParseTuple(args, "s|O", &cname, &create_if_missing)) {
        PyErr_SetString(PyExc_ValueError, "SwitchWorkspaceCC takes a workspace name and a optional "
                                          "bool value that specific whether to create the workspace if missing.");
        return nullptr;
    }
    bool success = SwitchWorkspaceInternal(string(cname), PyObject_IsTrue(create_if_missing));
    if (!success) {
        PyErr_SetString(PyExc_RuntimeError, "workspace of the given name is not exist and "
                                            "is not allowed to create. (try alllow ?)");
        return nullptr;
    }
    Py_RETURN_TRUE;
}

PyObject* CurrentWorkspaceCC(PyObject* self, PyObject* args) { 
    return StdStringToPyUnicode(g_current_workspace);
}

PyObject* WorkspacesCC(PyObject* self, PyObject* args) {
    PyObject* list = PyList_New(g_workspaces.size());
    int i = 0;
    for (auto const& it : g_workspaces) 
        CHECK_EQ(PyList_SetItem(list, i++, StdStringToPyUnicode(it.first)), 0);
    return list;
}

PyObject* ResetWorkspaceCC(PyObject* self, PyObject* args) {
    char* cname;
    if (!PyArg_ParseTuple(args, "|s", &cname)) {
        PyErr_SetString(PyExc_ValueError, "ResetWorkspaceCC takes in either no args or a string "
                                          "specifing the name of the workspace.");
        return nullptr;
    }
    LOG(INFO) << "Reset the Workspace(" << g_current_workspace << ")";
	string workspace_name = string(cname);
	if (workspace_name.empty()) g_workspaces[g_current_workspace].reset(new Workspace());
    else g_workspaces[g_current_workspace].reset(new Workspace(workspace_name));
    g_workspace = g_workspaces[g_current_workspace].get();
    Py_RETURN_TRUE;
}

PyObject* RootFolderCC(PyObject* self, PyObject* args) {
    return StdStringToPyUnicode(g_workspace->GetRootFolder());
}

PyObject* TensorsCC(PyObject* self, PyObject* args) {
    vector<string> tensor_strings = g_workspace->GetTensors();
    PyObject* list = PyList_New(tensor_strings.size());
    for (int i = 0; i < tensor_strings.size(); i++)
        CHECK_EQ(PyList_SetItem(list, i, StdStringToPyUnicode(tensor_strings[i])), 0);
    return list;
}

PyObject* CreateTensorCC(PyObject* self, PyObject* args) {
    char* cname;
    if (!PyArg_ParseTuple(args, "s", &cname)) {
        PyErr_SetString(PyExc_ValueError, "CreateTensorCC must accept a tensor name to create.");
        return nullptr;
    }
    g_workspace->CreateTensor(string(cname));
    Py_RETURN_TRUE;
}

PyObject* CreateFillerCC(PyObject* self, PyObject* args) {
    PyObject* filler_string;
    if (!PyArg_ParseTuple(args, "S", &filler_string)) {
        PyErr_SetString(PyExc_ValueError, "CreateFillerCC requires an input that is a serialized filler protobuf.");
        return nullptr;
    }
    TensorFiller filler_def;
    if (!filler_def.ParseFromString(PyBytesToStdString(filler_string))) {
        PyErr_SetString(PyExc_RuntimeError, "CreateFillerCC can't parse the filler.");
        return nullptr;
    }
    g_workspace->CreateFiller(filler_def);
    g_workspace->CreateTensor(filler_def.tensor());
    Py_RETURN_TRUE;
}

PyObject* HasTensorCC(PyObject* self, PyObject* args) {
    char* cname;
    if (!PyArg_ParseTuple(args, "s", &cname)) return nullptr;
    if (g_workspace->HasTensor(string(cname))) Py_RETURN_TRUE;
    else Py_RETURN_FALSE;
}

PyObject* GetTensorNameCC(PyObject* self, PyObject* args) {
    char* cname;
    if (!PyArg_ParseTuple(args, "s", &cname)) return nullptr;
    string query = g_workspace->GetTensorName(string(cname));
    return StdStringToPyUnicode(query);
}

PyObject* CreateGraphCC(PyObject* self, PyObject* args) {
    PyObject* graph_str;
    if (!PyArg_ParseTuple(args, "S", &graph_str)) {
        PyErr_SetString(PyExc_ValueError, "CreateGraphCC requires an input that is a serialized net protobuf.");
        return nullptr;
    }
    GraphDef graph_def;
    if (!graph_def.ParseFromString(PyBytesToStdString(graph_str))) {
        PyErr_SetString(PyExc_RuntimeError, "CreateGraphCC can not parse the net.");
        return nullptr;
    } 
    if (!g_workspace->CreateGraph(graph_def)) {
        PyErr_SetString(PyExc_RuntimeError, "CreateGraphCC can not create the net.");
        return nullptr;
    }
    Py_RETURN_TRUE;
}

PyObject* RunGraphCC(PyObject* self, PyObject* args) {
    char* cname, *include, *exclude;
    if (!PyArg_ParseTuple(args, "sss", &cname, &include, &exclude)) {
        PyErr_SetString(PyExc_ValueError, "RunGraphCC requires a net name and rules.");
        return nullptr;
    }
    bool result = g_workspace->RunGraph(string(cname), string(include), string(exclude));
    if (!result) { 
        PyErr_SetString(PyExc_RuntimeError, "RunGraphCC can' t run the net.");
        return nullptr;
    }
    Py_RETURN_TRUE;
}

PyObject* GraphsCC(PyObject* self, PyObject* args) {
    vector<string> graph_string = g_workspace->GetGraphs();
    PyObject* list = PyList_New(graph_string.size());
    for (int i = 0; i < graph_string.size(); i++)
        CHECK_EQ(PyList_SetItem(list, i, StdStringToPyUnicode(graph_string[i])), 0);
    return list;
}

PyObject* FetchTensorCC(PyObject* self, PyObject* args) {
    char* cname;
    if (!PyArg_ParseTuple(args, "s", &cname)) {
        PyErr_SetString(PyExc_ValueError, "FetchTensorCC must specify a tensor name to fetch.");
        return nullptr;
    }
    if (!g_workspace->HasTensor(string(cname))) {
        PyErr_SetString(PyExc_ValueError, "FetchTensorCC found tensor doesn't exist, try create it before.");
        return nullptr;
    }
    Tensor* tensor = g_workspace->GetTensor(string(cname));
    TypeId type_id = CTypeToFetcher(tensor->meta().id());
    CHECK(type_id != 0) << "\nTensor(" << tensor->name()
                        << ") has not been computed yet.";
    unique_ptr<TensorFetcherBase> fetcher(createFetcher(type_id));
    if (fetcher.get()) {
        return fetcher->Fetch(*tensor);  // copy the tensor data to a numpy object
    } else {
        LOG(INFO) << string(cname) << " is not a C++ native type.";
        return nullptr;
    }
}

PyObject* FeedTensorCC(PyObject* self, PyObject* args) {
    char* cname;
    PyArrayObject* array = nullptr;
    PyObject *device_option = nullptr;
    if (!PyArg_ParseTuple(args, "sO|O", &cname, &array, &device_option)) {
        PyErr_SetString(PyExc_ValueError, "FeedTensorCC accpets incorrect args.");
        return nullptr;
    }
    DeviceOption option;
    if (device_option != nullptr) {
        if (!option.ParseFromString(PyBytesToStdString(device_option))) {
            PyErr_SetString(PyExc_ValueError, "FeedTensorCC can't parse the device option.");
            return nullptr;
        }
    }
    Tensor* tensor = g_workspace->CreateTensor(string(cname));
    unique_ptr<TensorFeederBase> feeder(TensorFeederRegistry()->Create(TypeMeta::Id<NumpyFeeder>()));
    if (feeder.get()) {
        return feeder->Feed(option, array, tensor);
    } else {
        PyErr_SetString(PyExc_TypeError, "FeedTensorCC encounters the unknown device type.");
        return nullptr;
    }
}

PyObject* RestoreCC(PyObject* self, PyObject* args) {
    char* cname, *namescope;
    int format;
    if (!PyArg_ParseTuple(args, "ssi", &cname, &namescope, &format)) {
        PyErr_SetString(PyExc_ValueError, "RestoreCC accpets incorrect args.");
        return nullptr;
    }
    switch (format) {
        case 0:  // cPickle
            PyErr_SetString(PyExc_NotImplementedError, "format(0) depends on cPickle, should not be used in CC.");
            break;
        case 1:    // caffemodel
            LoadCaffeModel(string(cname), string(namescope), g_workspace);
            break;
        default: LOG(FATAL) << "unknwon Restore Format, code: " << format;
    }
    Py_RETURN_TRUE;
}

PyObject* SnapshotCC(PyObject* self, PyObject* args) {
    char* cname;
    int format;
    PyObject* names; vector<Tensor*> tensors;
    if (!PyArg_ParseTuple(args, "sOi", &cname, &names, &format)) {
        PyErr_SetString(PyExc_ValueError, "SnapshotCC accpets incorrect args.");
        return nullptr;
    }
    switch (format) {
        case 0:    //  cPickle
            PyErr_SetString(PyExc_NotImplementedError, "format(0) depends on cPickle, should not be used in CC.");
            break;
        case 1:    //  caffemodel
            for (int i = 0; i < PyList_Size(names); i++)
                tensors.push_back(g_workspace->GetTensor(PyString_AsString(PyList_GetItem(names, i))));
            SavaCaffeModel(string(cname), tensors);
            break;
        default: LOG(FATAL) << "Unknwon Restore Format, code: " << format;
   }
   Py_RETURN_TRUE;
}

PyObject* SetLogLevelCC(PyObject* self, PyObject* args) {
    char* cname;
    if (!PyArg_ParseTuple(args, "s", &cname)) {
        PyErr_SetString(PyExc_ValueError, "SetLogLevelCC accpets a str of DEBUG, or "
                                          "INFO, WARNING, ERROR and FATAL.");
        return nullptr;
    }
    SetLogDestination(StrToLogSeverity(string(cname)));
    Py_RETURN_TRUE;
}

#define PYFUNC(name) {#name, name, METH_VARARGS, ""}
#define PYENDFUNC {nullptr, nullptr, 0, nullptr}

PyMethodDef* GetAllMethods() {
    static PyMethodDef g_python_methods[] {
        PYFUNC(RegisteredOperatorsCC),
        PYFUNC(NoGradientOperatorsCC),
        PYFUNC(CreateGradientDefsCC),
        PYFUNC(SwitchWorkspaceCC),
        PYFUNC(CurrentWorkspaceCC),
        PYFUNC(WorkspacesCC),
        PYFUNC(ResetWorkspaceCC),
        PYFUNC(RootFolderCC),
        PYFUNC(TensorsCC),
        PYFUNC(HasTensorCC),
        PYFUNC(GetTensorNameCC),
        PYFUNC(CreateGraphCC),
        PYFUNC(RunGraphCC),
        PYFUNC(GraphsCC),
        PYFUNC(CreateTensorCC),
        PYFUNC(CreateFillerCC),
        PYFUNC(FetchTensorCC),
        PYFUNC(FeedTensorCC),
        PYFUNC(RestoreCC),
        PYFUNC(SnapshotCC),
        PYFUNC(SetLogLevelCC),
        PYFUNC(MPIInitCC),
        PYFUNC(MPIRankCC),
        PYFUNC(MPISizeCC),
        PYFUNC(MPICreateGroupCC),
        PYFUNC(MPIFinalizeCC),
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