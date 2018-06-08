//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

/*Licensed under the Apache License, Version 2.0(the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. */

#include "mkldnn.hpp"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/operators/sum_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::CPUDeviceContext;
using framework::DataLayout;

template <typename T>
class SumMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto in_vars = ctx.MultiInputVar("X");

    const int N = in_vars.size();

    const std::string key = ctx.op().Output("Out");
    auto out_var = ctx.OutputVar("Out");
    auto out_vars = ctx.Output<Tensor>("Out");
    bool in_place = out_var == in_vars[0];

    if (out_var->IsType<framework::LoDTensor>()) {
      auto* out = ctx.Output<framework::LoDTensor>("Out");

      std::vector<float> scale;
      std::vector<mkldnn::memory> srcs;
      std::vector<mkldnn::memory::primitive_desc> srcs_pd;

      T* output_data = out->mutable_data<T>(ctx.GetPlace());

      math::SelectedRowsAddToTensor<CPUDeviceContext, T> functor;

      std::vector<int> dims;
      for (int i = in_place ? 1 : 0; i < N; i++) {
        if (in_vars[i]->IsType<framework::LoDTensor>()) {
          auto& dtype = in_vars[i]->Get<framework::LoDTensor>();

          if (dtype.numel() == 0) {
            continue;
          }

          dims = paddle::framework::vectorize2int(dtype.dims());
          const T* input_data = dtype.data<T>();

          auto src_md =
              dims.size() == 4
                  ? platform::MKLDNNMemDesc(
                        dims, mkldnn::memory::data_type::f32, dtype.format())
                  : platform::MKLDNNMemDesc(dims,
                                            mkldnn::memory::data_type::f32,
                                            mkldnn::memory::format::nc);

          auto src_pd = mkldnn::memory::primitive_desc(src_md, mkldnn_engine);

          auto src_memory = mkldnn::memory({src_md, mkldnn_engine},
                                           platform::to_void_cast(input_data));

          srcs_pd.push_back(src_pd);
          srcs.push_back(src_memory);
          scale.push_back(1.0);
        } else if (in_vars[i]->IsType<framework::SelectedRows>()) {
          auto& in_t = in_vars[i]->Get<framework::SelectedRows>();
          functor(ctx.template device_context<CPUDeviceContext>(), in_t, out);
        } else {
          PADDLE_THROW("Variable type must be LoDTensor/SelectedRows.");
        }
      }
      if (srcs_pd.size() != 0) {
        auto& out_lod = out_var->Get<framework::LoDTensor>();

        std::vector<int> dims_out =
            paddle::framework::vectorize2int(out_lod.dims());

        auto dst_md =
            platform::MKLDNNMemDesc(dims_out, mkldnn::memory::data_type::f32,
                                    mkldnn::memory::format::any);

        auto sum_pd = mkldnn::sum::primitive_desc(dst_md, scale, srcs_pd);

        auto dst_memory =
            mkldnn::memory(sum_pd.dst_primitive_desc(), output_data);

        std::vector<mkldnn::primitive::at> inputs;

        for (int j = 0; j < N; ++j) {
          inputs.push_back(srcs[j]);
        }

        auto c = mkldnn::sum(sum_pd, inputs, dst_memory);

        std::vector<mkldnn::primitive> pipeline = {c};
        mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

        out_vars->set_layout(DataLayout::kMKLDNN);
        out_vars->set_format(
            (mkldnn::memory::format)dst_memory.get_primitive_desc()
                .desc()
                .data.format);
      }
    } else if (out_var->IsType<framework::SelectedRows>()) {
      // TODO(@mozga-intel) Add MKLDNN SelectedRows support
      std::unique_ptr<framework::SelectedRows> in0;
      if (in_place) {
        // If is in_place, we store the input[0] to in0
        auto& in_sel0 = in_vars[0]->Get<SelectedRows>();
        auto& rows = in_sel0.rows();
        in0.reset(new framework::SelectedRows(rows, in_sel0.height()));
        in0->mutable_value()->ShareDataWith(in_sel0.value());
      }

      auto get_selected_row = [&](size_t i) -> const SelectedRows& {
        if (i == 0 && in0) {
          return *in0.get();
        } else {
          return in_vars[i]->Get<SelectedRows>();
        }
      };
      auto* out = ctx.Output<SelectedRows>("Out");
      out->mutable_rows()->clear();
      auto* out_value = out->mutable_value();

      // Runtime InferShape
      size_t first_dim = 0;
      for (int i = 0; i < N; i++) {
        auto& sel_row = get_selected_row(i);
        first_dim += sel_row.rows().size();
      }
      auto in_dim =
          framework::vectorize(get_selected_row(N - 1).value().dims());
      in_dim[0] = static_cast<int64_t>(first_dim);

      out_value->Resize(framework::make_ddim(in_dim));

      // if all the input sparse vars are empty, no need to
      // merge these vars.
      if (first_dim == 0UL) {
        return;
      }
      out_value->mutable_data<T>(ctx.GetPlace());
      math::SelectedRowsAddTo<CPUDeviceContext, T> functor;
      int64_t offset = 0;
      for (int i = 0; i < N; i++) {
        auto& sel_row = get_selected_row(i);
        if (sel_row.rows().size() == 0) {
          continue;
        }
        PADDLE_ENFORCE_EQ(out->height(), sel_row.height());
        functor(ctx.template device_context<CPUDeviceContext>(), sel_row,
                offset, out);
        offset += sel_row.value().numel();
      }
    } else if (out_var->IsType<framework::LoDTensorArray>()) {
      // TODO(@mozga-intel) Add MKLDNN LoDTensorArray support
      auto& out_array = *out_var->GetMutable<framework::LoDTensorArray>();
      for (size_t i = in_place ? 1 : 0; i < in_vars.size(); ++i) {
        PADDLE_ENFORCE(in_vars[i]->IsType<framework::LoDTensorArray>(),
                       "Only support all inputs are TensorArray");
        auto& in_array = in_vars[i]->Get<framework::LoDTensorArray>();

        for (size_t i = 0; i < in_array.size(); ++i) {
          if (in_array[i].numel() != 0) {
            if (i >= out_array.size()) {
              out_array.resize(i + 1);
            }
            if (out_array[i].numel() == 0) {
              framework::TensorCopy(in_array[i], in_array[i].place(),
                                    ctx.device_context(), &out_array[i]);
              out_array[i].set_lod(in_array[i].lod());
            } else {
              PADDLE_ENFORCE(out_array[i].lod() == in_array[i].lod());
              auto in = EigenVector<T>::Flatten(in_array[i]);
              auto result = EigenVector<T>::Flatten(out_array[i]);
              result.device(*ctx.template device_context<MKLDNNDeviceContext>()
                                 .eigen_device()) = result + in;
            }
          }
        }
      }
    } else {
      PADDLE_THROW("Unexpected branch, output variable type is %s",
                   out_var->Type().name());
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_KERNEL(sum, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::SumMKLDNNOpKernel<float>);
