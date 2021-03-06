//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// #ifndef _WIN32
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/platform/collective_helper.h"

#include <functional>

#include "paddle/fluid/platform/dynload/nccl.h"

namespace paddle {
namespace platform {

class NCCLCommImpl : public NCCLComm {
 public:
  void set_ring_id(int ring_id) { ring_id_ = ring_id; }
  int ring_id() const override { return ring_id_; }

  void set_nranks(int nranks) { nranks_ = nranks; }
  int nranks() const override { return nranks_; }

  void set_rank(int rank) { rank_ = rank; }
  int rank() const override { return rank_; }

  void set_local_rank(int local_rank) { local_rank_ = local_rank; }
  int local_rank() const override { return local_rank_; }

  void set_comm(ncclComm_t comm) { comm_ = comm; }
  ncclComm_t comm() const override { return comm_; }

  void set_dev_ctx(CUDADeviceContext* dev_ctx) { dev_ctx_ = dev_ctx; }
  CUDADeviceContext* DevCtx() const override { return dev_ctx_; }

  cudaStream_t stream() const override { return dev_ctx_->stream(); }

 private:
  int ring_id_;
  int nranks_;
  int rank_;
  int local_rank_;
  ncclComm_t comm_;
  CUDADeviceContext* dev_ctx_;
};

// NOTE: not thread-safe
NCCLComm* NCCLCommContext::CreateNCCLComm(ncclUniqueId* nccl_id, int nranks,
                                          int rank, int dev_id, int ring_id) {
  PADDLE_ENFORCE_NOT_NULL(nccl_id);
  PADDLE_ENFORCE_GT(nranks, 1);
  PADDLE_ENFORCE(rank >= 0 && rank < nranks,
                 "Expected rank id range [0, %d), but get %d", nranks, rank);
  PADDLE_ENFORCE_GE(dev_id, 0);

  if (dev_ctx_map_.count(dev_id) == 0) {
    dev_ctx_map_.emplace(dev_id, std::unique_ptr<CUDADeviceContext>(
                                     new CUDADeviceContext(CUDAPlace(dev_id))));
  }

  ncclComm_t comm = nullptr;
  PADDLE_ENFORCE(cudaSetDevice(dev_id));
  PADDLE_ENFORCE(
      platform::dynload::ncclCommInitRank(&comm, nranks, *nccl_id, rank));

  NCCLCommImpl* communicator = new NCCLCommImpl;
  communicator->set_ring_id(ring_id);
  communicator->set_nranks(nranks);
  communicator->set_rank(rank);
  communicator->set_local_rank(dev_id);
  communicator->set_comm(comm);
  communicator->set_dev_ctx(dev_ctx_map_.at(dev_id).get());

  comm_map_.emplace(ring_id, std::unique_ptr<NCCLComm>(communicator));

  VLOG(0) << "nccl communicator of rank " << rank << " in ring " << ring_id
          << " has been created";

  return comm_map_.at(ring_id).get();
}

NCCLCommContext::~NCCLCommContext() {
  for (auto& p : comm_map_) {
    PADDLE_ENFORCE(platform::dynload::ncclCommDestroy(p.second->comm()));
  }
}

}  // namespace platform
}  // namespace paddle

#endif
