/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/timer.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_bool(profile, false, "Turn on profiler for fluid");

namespace paddle {
namespace {
using paddle::inference::Timer;

template <class T>
std::string num2str(T a) {
  std::stringstream istr;
  istr << a;
  return istr.str();
}
}  // namespace

void NativePaddlePredictor::PrepareFeedFetch() {
  for (auto *op : inference_program_->Block(0).AllOps()) {
    if (op->Type() == "feed") {
      int idx = boost::get<int>(op->GetAttr("col"));
      if (feeds_.size() <= static_cast<size_t>(idx)) {
        feeds_.resize(idx + 1);
      }
      feeds_[idx] = op;
      feed_names_[op->Output("Out")[0]] = idx;
    } else if (op->Type() == "fetch") {
      int idx = boost::get<int>(op->GetAttr("col"));
      if (fetchs_.size() <= static_cast<size_t>(idx)) {
        fetchs_.resize(idx + 1);
      }
      fetchs_[idx] = op;
    }
  }
}

bool NativePaddlePredictor::Init(
    std::shared_ptr<framework::Scope> parent_scope) {
  VLOG(3) << "Predictor::init()";
#if !defined(_WIN32)
  if (FLAGS_profile) {
    LOG(WARNING) << "Profiler is actived, might affect the performance";
    LOG(INFO) << "You can turn off by set gflags '-profile false'";

    auto tracking_device = config_.use_gpu ? platform::ProfilerState::kAll
                                           : platform::ProfilerState::kCPU;
    platform::EnableProfiler(tracking_device);
  }
#endif

  if (config_.use_gpu) {
    place_ = paddle::platform::CUDAPlace(config_.device);
  } else {
    place_ = paddle::platform::CPUPlace();
  }
  if (parent_scope) {
    scope_ = parent_scope;
    sub_scope_ = &(parent_scope->NewScope());
    PADDLE_ENFORCE_NOT_NULL(sub_scope_, "create sub scope fail");
  } else {
    paddle::framework::InitDevices(false);
    scope_.reset(new paddle::framework::Scope());
  }

  executor_.reset(new paddle::framework::Executor(place_));

  // Initialize the inference program
  if (!config_.model_dir.empty()) {
    // Parameters are saved in separate files sited in
    // the specified `dirname`.
    inference_program_ = paddle::inference::Load(executor_.get(), scope_.get(),
                                                 config_.model_dir);
  } else if (!config_.prog_file.empty() && !config_.param_file.empty()) {
    // All parameters are saved in a single file.
    // The file names should be consistent with that used
    // in Python API `fluid.io.save_inference_model`.
    inference_program_ = paddle::inference::Load(
        executor_.get(), scope_.get(), config_.prog_file, config_.param_file);
  } else {
    LOG(ERROR) << "fail to load inference model.";
    return false;
  }

  ctx_ = executor_->Prepare(*inference_program_, 0);
  if (config_._use_mkldnn) {
    executor_->EnableMKLDNN(*inference_program_);
  }
  executor_->CreateVariables(*inference_program_,
                             sub_scope_ ? sub_scope_ : scope_.get(), 0);

  // Get the feed_target_names and fetch_target_names
  PrepareFeedFetch();
  return true;
}

NativePaddlePredictor::~NativePaddlePredictor() {
#if !defined(_WIN32)
  if (FLAGS_profile) {
    platform::DisableProfiler(platform::EventSortingKey::kTotal,
                              "./profile.log");
  }
#endif
  if (sub_scope_) {
    scope_->DeleteScope(sub_scope_);
  }
}

bool NativePaddlePredictor::Run(const std::vector<PaddleTensor> &inputs,
                                std::vector<PaddleTensor> *output_data,
                                int batch_size) {
  VLOG(3) << "Predictor::predict";
  Timer timer;
  timer.tic();
  // set feed variable
  std::vector<framework::LoDTensor> feeds;
  framework::Scope *scope = sub_scope_ != nullptr ? sub_scope_ : scope_.get();
  if (!SetFeed(inputs, scope)) {
    LOG(ERROR) << "fail to set feed";
    return false;
  }
  // Run the inference program
  // if share variables, we need not create variables
  VLOG(4) << "Run prepared context";
  executor_->RunPreparedContext(ctx_.get(), scope,
                                false, /* don't create local scope each time*/
                                false /* don't create variable eatch time */);
  VLOG(4) << "Finish prepared context";
  // get fetch variable
  if (!GetFetch(output_data, scope)) {
    LOG(ERROR) << "fail to get fetches";
    return false;
  }
  VLOG(3) << "predict cost: " << timer.toc() << "ms";
  return true;
}

std::unique_ptr<PaddlePredictor> NativePaddlePredictor::Clone() {
  VLOG(3) << "Predictor::clone";
  std::unique_ptr<PaddlePredictor> cls(new NativePaddlePredictor(config_));

  if (!dynamic_cast<NativePaddlePredictor *>(cls.get())->Init(scope_)) {
    LOG(ERROR) << "fail to call Init";
    return nullptr;
  }
#ifdef __clang__
  // fix clang compile error
  return cls;
#else
  // fix manylinux compile error.
  return std::move(cls);
#endif
}

bool NativePaddlePredictor::SetFeed(const std::vector<PaddleTensor> &inputs,
                                    framework::Scope *scope) {
  VLOG(3) << "Predictor::set_feed";
  if (inputs.size() != feeds_.size()) {
    LOG(ERROR) << "wrong feed input size, need " << feeds_.size() << " but get "
               << inputs.size();
    return false;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    framework::LoDTensor input;
    framework::DDim ddim = framework::make_ddim(inputs[i].shape);
    void *handler = inputs[i].data.data();
    if (inputs[i].dtype == PaddleDType::INT64) {
      input.fill_data<int64_t>(ddim, platform::CPUPlace(), handler);
    } else if (inputs[i].dtype == PaddleDType::FLOAT32) {
      input.fill_data<float>(ddim, platform::CPUPlace(), handler);
    } else {
      LOG(ERROR) << "unsupported feed type " << inputs[i].dtype;
      return false;
    }

    // TODO(Superjomn) Low performance, need optimization for heavy LoD copy.
    framework::LoD lod;
    for (auto &level : inputs[i].lod) {
      lod.emplace_back(level);
    }
    input.set_lod(lod);
    int idx = -1;
    if (config_.specify_input_name) {
      idx = feed_names_[inputs[i].name];
    } else {
      idx = boost::get<int>(feeds_[i]->GetAttr("col"));
    }
    framework::SetFeedVariable(scope, input, "feed", idx);
  }
  return true;
}
template <typename T>
void NativePaddlePredictor::GetFetchOne(const framework::LoDTensor &fetch,
                                        PaddleTensor *output) {
  std::vector<int> shape;
  auto dims_i = fetch.dims();
  auto lod = fetch.lod();
  const T *output_ptr = fetch.data<T>();
  auto num = fetch.numel();
  std::vector<T> data;
  if (0 == lod.size()) {
    std::copy(output_ptr, output_ptr + num, std::back_inserter(data));
    for (int j = 0; j < dims_i.size(); ++j) {
      shape.push_back(dims_i[j]);
    }
  } else {
    // for batch detection
    // image[0] -> output[0] shape {145, 6}
    // image[1] -> output[1] shape {176, 6}
    // then,
    // the batch output shape {321, 6}
    // the lod {{0, 145, 321}}
    // so we should append output[0] to {176, 6}
    size_t max_dim = 0;
    for (size_t j = 1; j < lod[0].size(); j++) {
      max_dim = std::max(max_dim, lod[0][j] - lod[0][j - 1]);
    }
    size_t common_dim = lod[0].back() == 0 ? 0 : num / lod[0].back();
    if (max_dim > 0) {
      data.resize((lod[0].size() - 1) * max_dim * common_dim, 0);
    }
    for (size_t j = 1; j < lod[0].size(); j++) {
      size_t start = lod[0][j - 1] * common_dim;
      size_t end = lod[0][j] * common_dim;
      if (end > start) {
        std::copy(output_ptr + start, output_ptr + end,
                  data.begin() + (j - 1) * max_dim * common_dim);
      }
    }
    shape.push_back(lod[0].size() - 1);
    shape.push_back(max_dim);
    for (int j = 1; j < dims_i.size(); ++j) {
      shape.push_back(dims_i[j]);
    }
  }

  output->shape = shape;
  auto &buffer = output->data;
  if (buffer.empty() || buffer.length() < sizeof(T) * data.size()) {
    buffer.Resize(sizeof(T) * data.size());
  }
  std::memcpy(buffer.data(), data.data(), sizeof(T) * data.size());
  // copy LoD
  for (const auto &level : fetch.lod()) {
    output->lod.emplace_back(level);
  }
}

bool NativePaddlePredictor::GetFetch(std::vector<PaddleTensor> *outputs,
                                     framework::Scope *scope) {
  VLOG(3) << "Predictor::get_fetch";
  outputs->resize(fetchs_.size());
  for (size_t i = 0; i < fetchs_.size(); ++i) {
    int idx = boost::get<int>(fetchs_[i]->GetAttr("col"));
    PADDLE_ENFORCE((size_t)idx == i);
    framework::LoDTensor &fetch =
        framework::GetFetchVariable(*scope, "fetch", idx);
    auto type = fetch.type();
    auto output = &(outputs->at(i));
    if (type == typeid(float)) {
      GetFetchOne<float>(fetch, output);
      output->dtype = PaddleDType::FLOAT32;
    } else if (type == typeid(int64_t)) {
      GetFetchOne<int64_t>(fetch, output);
      output->dtype = PaddleDType::INT64;
    } else {
      LOG(ERROR) << "unknown type, only support float32 and int64 now.";
    }
  }
  return true;
}

template <>
std::unique_ptr<PaddlePredictor> CreatePaddlePredictor<
    NativeConfig, PaddleEngineKind::kNative>(const NativeConfig &config) {
  VLOG(3) << "create NativePaddlePredictor";
  if (config.use_gpu) {
    // 1. GPU memeroy
    PADDLE_ENFORCE_GT(
        config.fraction_of_gpu_memory, 0.f,
        "fraction_of_gpu_memory in the config should be set to range (0., 1.]");
    PADDLE_ENFORCE_GE(config.device, 0, "Invalid device id %d", config.device);
    std::vector<std::string> flags;
    if (config.fraction_of_gpu_memory >= 0.0f ||
        config.fraction_of_gpu_memory <= 0.95f) {
      flags.push_back("dummpy");
      std::string flag = "--fraction_of_gpu_memory_to_use=" +
                         num2str<float>(config.fraction_of_gpu_memory);
      flags.push_back(flag);
      VLOG(3) << "set flag: " << flag;
      framework::InitGflags(flags);
    }
  }

  std::unique_ptr<PaddlePredictor> predictor(new NativePaddlePredictor(config));
  if (!dynamic_cast<NativePaddlePredictor *>(predictor.get())->Init(nullptr)) {
    return nullptr;
  }
#ifdef __clang__
  // fix clang compile error
  return predictor;
#else
  return std::move(predictor);
#endif
}

}  // namespace paddle
