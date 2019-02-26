/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

extern "C" {
#include <xxhash.h>
}
#include <libdivide.h>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
// template <typename DeviceContext, typename T>
template <typename T>
class HashKerel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* out_t = context.Output<framework::LoDTensor>("Out");
    auto* in_t = context.Input<framework::LoDTensor>("X");
    int mod_by = context.Attr<int>("mod_by");
    int num_hash = context.Attr<int>("num_hash");
    auto* output = out_t->mutable_data<T>(context.GetPlace());

    auto in_dims = in_t->dims();
    auto in_lod = in_t->lod();
    PADDLE_ENFORCE_EQ(
        static_cast<uint64_t>(in_dims[0]), in_lod[0].back(),
        "The actual input data's size mismatched with LoD information.");

    auto seq_length = in_dims[0];
    auto last_dim = in_dims[in_dims.size() - 1];
    auto* input = in_t->data<T>();
    const int buf_len = sizeof(int) * last_dim;
    libdivide::divider<T> fast_d(static_cast<T>(mod_by));
    for (int idx = 0; idx < seq_length; ++idx) {
      for (int ihash = 0; ihash < num_hash; ihash += 4) {
        __m256i vhash;
        T h1, h2, h3, h4;
        h1 = XXH64(input, buf_len, ihash);
        h2 = XXH64(input, buf_len, ihash + 1);
        h3 = XXH64(input, buf_len, ihash + 2);
        h4 = XXH64(input, buf_len, ihash + 3);
        h1 = h1 - (h1 / fast_d) * mod_by;
        h2 = h2 - (h2 / fast_d) * mod_by;
        h3 = h3 - (h3 / fast_d) * mod_by;
        h4 = h4 - (h4 / fast_d) * mod_by;
        vhash = _mm256_set_epi64x(h4, h3, h2, h1);
        _mm256_stream_si256((__m256i*)&output[idx * num_hash + ihash],
                            (__m256i)vhash);
      }
      input += last_dim;
    }
  }
};

}  // namespace operators
}  // namespace paddle
