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

#ifndef DRAGON_UTILS_CAFFEMODEL_H_
#define DRAGON_UTILS_CAFFEMODEL_H_

#include "core/workspace.h"
#include "proto/caffemodel.pb.h"

namespace dragon {

inline void LoadCaffeModel(
    string                          file,
    Workspace*                      ws) {
    NetParameter net_param;
    ReadProtoFromBinaryFile(file.c_str(), &net_param);
    LOG(INFO) << "Restore From Model @: " << file << "......";
    LOG(INFO) << "Model Format: CaffeModel";
    for (int i = 0; i < net_param.layer_size(); i++) {
        const auto& layer = net_param.layer(i);
        const auto& layer_name = layer.name();
        auto prefix = layer_name + "/param:";
        for (int j = 0; j < layer.blobs_size(); j++) {
            auto tensor_name = prefix + str::to(j);
            if (!ws->HasTensor(tensor_name)) {
                LOG(WARNING)
                    << "Tensor(" << tensor_name << ") "
                    << "does not exist in any Graphs, skip.";
            } else {
                auto blob = layer.blobs(j);
                vec64_t tensor_shape;
                for (auto dim : blob.shape().dim())
                    tensor_shape.push_back(dim);
                auto* tensor = ws->GetTensor(tensor_name);
                std::stringstream DimString;
                if (tensor_shape.size() > 0) {
                    tensor->Reshape(tensor_shape);
                    CHECK_EQ(tensor->count(), blob.data_size())
                        << "\nTensor(" << tensor_name << ") "
                        << "failed to load, except size:  "
                        << tensor->count()
                        << ", loaded: " << blob.data_size();
                    DimString << tensor->DimString();
                } else {
                    tensor->Reshape({ blob.data_size() });
                    DimString << "(missing)";
                }
                auto* x = tensor->mutable_data<float, CPUContext>();
                for (int xi = 0; xi < blob.data_size(); ++xi)
                    x[xi] = blob.data(xi);
                LOG(INFO) << "Tensor(" << tensor_name << ") "
                          << "loaded, shape: " << DimString.str()
                          << ", size: " << blob.data_size();
            }
        }
    }
}

inline void SavaCaffeModel(
    string                          file,
    const vector<Tensor*>&          tensors) {
    int j = -1;
    NetParameter net;
    Map<string, int> layer_hash;
    for (int i = 0; i < tensors.size(); i++) {
        if (tensors[i]->count() <= 0) continue;
        auto splits = str::split(
            tensors[i]->name(), "/param:");
        if (layer_hash.count(splits[0]) == 0) {
            layer_hash[splits[0]] = ++j;
            auto* layer = net.add_layer();
            layer->set_name(splits[0]);
        }
        auto* blob = net.mutable_layer(j)->add_blobs();
        for (auto dim : tensors[i]->dims())
            blob->mutable_shape()->add_dim(dim);
        if (XIsType((*tensors[i]), float)) {
            auto* x = tensors[i]->data<float, CPUContext>();
            for (int xi = 0; xi < tensors[i]->count(); ++xi)
                blob->mutable_data()->Add(x[xi]);
        } else if (XIsType((*tensors[i]), float16)) {
            auto* x = tensors[i]->data<float16, CPUContext>();
            for (int xi = 0; xi < tensors[i]->count(); ++xi)
                blob->mutable_data()->Add(
                    cast::to<float>(x[xi]));
        }
    }
    WriteProtoToBinaryFile(net, file.c_str());
    LOG(INFO) << "Save the model @: " << file << "......";
    LOG(INFO) << "Model format: Caffe";
}

}  // namespace dragon

#endif  // DRAGON_UTILS_CAFFEMODEL_H_