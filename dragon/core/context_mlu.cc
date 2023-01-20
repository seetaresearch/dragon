#ifdef USE_MLU

#include "dragon/core/context_mlu.h"
#include "dragon/core/workspace.h"

namespace dragon {

MLUObjects::~MLUObjects() {
  for (int device_id = 0; device_id < MLU_MAX_DEVICES; ++device_id) {
    for (auto& iter : cncl_comms_[device_id]) {
      if (iter.second) cnclDestroyComms(&iter.second, 1);
    }
    for (auto& iter : cnrand_generators_[device_id]) {
      if (iter.second) cnnlRandDestroyGenerator(iter.second);
    }
    for (auto& handle : cnnl_handles_[device_id]) {
      if (handle) cnnlDestroy(handle);
    }
    for (auto& stream : streams_[device_id]) {
      if (stream) cnrtQueueDestroy(stream);
    }
    for (auto& workspace : workspaces_[device_id]) {
      if (workspace) delete workspace;
    }
  }
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
