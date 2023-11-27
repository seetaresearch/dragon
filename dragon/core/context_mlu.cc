#ifdef USE_MLU

#include "dragon/core/context_mlu.h"
#include "dragon/core/workspace.h"

namespace dragon {

MLUObjects::~MLUObjects() {
  for (int device_id = 0; device_id < MLU_MAX_DEVICES; ++device_id) {
    for (auto& iter : cncl_comms_[device_id]) {
      if (iter.second) CNCL_CHECK(cnclDestroyComms(&iter.second, 1));
    }
    for (auto& iter : cnrand_generators_[device_id]) {
      if (iter.second) CNNL_CHECK(cnnlRandDestroyGenerator(iter.second));
    }
    for (auto& handle : cnnl_handles_[device_id]) {
      if (handle) CNNL_CHECK(cnnlDestroy(handle));
    }
    for (auto& stream : streams_[device_id]) {
      if (stream) CNRT_CHECK(cnrtQueueDestroy(stream));
    }
    for (auto& workspace : workspaces_[device_id]) {
      if (workspace) delete workspace;
    }
  }
}

cnrtQueue_t MLUObjects::stream(int device_id, int stream_id) {
  auto& streams = streams_[device_id];
  if (streams.size() <= unsigned(stream_id)) {
    streams.resize(stream_id + 1, nullptr);
  }
  if (!streams[stream_id]) {
    MLUDeviceGuard guard(device_id);
    CNRT_CHECK(cnrtQueueCreate(&streams[stream_id]));
  }
  return streams[stream_id];
}

cnnlHandle_t MLUObjects::cnnl_handle(int device_id, int stream_id) {
  auto& handles = cnnl_handles_[device_id];
  if (handles.size() <= (unsigned)stream_id) {
    handles.resize(stream_id + 1, nullptr);
  }
  if (!handles[stream_id]) {
    MLUDeviceGuard guard(device_id);
    CNNL_CHECK(cnnlCreate(&handles[stream_id]));
    auto& handle = handles[stream_id];
    CNNL_CHECK(cnnlSetQueue(handle, stream(device_id, stream_id)));
  }
  return handles[stream_id];
}

Workspace* MLUObjects::workspace(int device_id, int stream_id) {
  auto& workspaces = workspaces_[device_id];
  if (workspaces.size() <= unsigned(stream_id)) {
    workspaces.resize(stream_id + 1, nullptr);
  }
  if (!workspaces[stream_id]) {
    workspaces[stream_id] = new Workspace("");
  }
  return workspaces[stream_id];
}

std::pair<cnnlRandGenerator_t, void*>
MLUObjects::cnrand_generator(int device_id, int stream_id, int seed) {
  auto& generators = cnrand_generators_[device_id];
  const string key = str::to(stream_id) + "/RNGState:" + str::to(seed);
  auto* ws = workspace(device_id, stream_id);
  auto find_iter = generators.find(key);
  if (find_iter != generators.end()) {
    void* state = ws->GetTensor(key)->raw_mutable_data<MLUContext>();
    return std::make_pair(find_iter->second, state);
  }
  MLUDeviceGuard guard(device_id);
  CNNL_CHECK(cnnlRandCreateGenerator(&generators[key], CNNL_RAND_RNG_MTGP32));
  auto& generator = generators[key];
  auto handle = cnnl_handle(device_id, stream_id);
  CNNL_CHECK(cnnlRandSetPseudoRandomGeneratorSeed(generator, seed));
  size_t state_size = 0, param_size = 0;
  cnnlMTGP32FastParams_t desc;
  CNNL_CHECK(cnnlRandGetMTGP32StateSize(generator, &state_size));
  CNNL_CHECK(cnnlRandGetMTGP32KernelParamSize(generator, &param_size));
  CNNL_CHECK(cnnlRandGetMTGP32HostParam(generator, &desc));
  auto* state = ws->CreateTensor(key)
                    ->Reshape({int64_t(state_size + param_size)})
                    ->mutable_data<uint8_t, MLUContext>();
  auto* params = state + state_size;
  CNNL_CHECK(cnnlRandMakeMTGP32Constants(handle, desc, params));
  CNNL_CHECK(cnnlRandMakeMTGP32KernelState(handle, state, desc, params, seed));
  return std::make_pair(generator, (void*)state);
}

cnclComm_t MLUObjects::cncl_comm(
    int device_id,
    const string& cache_key,
    cnclCliqueId* comm_uuid,
    int comm_size,
    int comm_rank) {
  auto& comms = cncl_comms_[device_id];
  auto find_iter = comms.find(cache_key);
  if (find_iter != comms.end()) return find_iter->second;
  if (comm_uuid == nullptr) return nullptr;
  MLUDeviceGuard guard(device_id);
  CNCL_CHECK(cnclInitComms(
      &comms[cache_key], 1, &device_id, &comm_rank, comm_size, comm_uuid));
  return comms[cache_key];
}

std::mutex& MLUContext::mutex() {
  static std::mutex m;
  return m;
}

MLUObjects& MLUContext::objects() {
  static thread_local MLUObjects objects_;
  return objects_;
}

} // namespace dragon

#endif // USE_MLU
