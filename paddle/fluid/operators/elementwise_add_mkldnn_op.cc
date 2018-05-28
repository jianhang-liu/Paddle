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

#include "paddle/fluid/operators/elementwise_add_op.h"
#include "paddle/fluid/memory/memcpy.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using mkldnn::memory;
using mkldnn::reorder;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::sum;

template <typename T>
inline void *cast_const_to_void(const T *t) {
  return static_cast<void *>(const_cast<T *>(t));
}

template <typename DeviceContext, typename T>
class EltwiseAddMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    const T* x_data = x->data<T>();
    const T* y_data = y->data<T>();
    T* z_data = z->mutable_data<T>(ctx.GetPlace());
    //  int axis = ctx.Attr<int>("axis");
    auto x_dims = x->dims();
    auto y_dims = y->dims();
    auto z_dims = z->dims();

    // element_wise OP should support dimension broadcast
    // since MKLDNN sum primitive doesn't support it, broadcast
    // need be done here before calling MKLDNN sum primitive
    // Will do this later
    if (x_dims.size() != y_dims.size())
      PADDLE_ENFORCE(x_dims.size() == y_dims.size(),
        "Only support inputs with same dimensions now");

    PADDLE_ENFORCE(x->layout() == DataLayout::kMKLDNN &&
                   x->format() != memory::format::format_undef,
                   "Wrong layout/format set for X tensor");
    PADDLE_ENFORCE(y->layout() == DataLayout::kMKLDNN &&
                   y->format() != memory::format::format_undef,
                   "Wrong layout/format set for X tensor");

    std::vector<int> src_x_tz = framework::vectorize2int(x_dims);
    std::vector<int> src_y_tz = framework::vectorize2int(y_dims);
    std::vector<int> dst_tz = framework::vectorize2int(z_dims);

    std::vector<memory::primitive_desc> srcs_pd;
    std::vector<memory> srcs;
    std::vector<float> scales = {1.0f, 1.0f};

    auto src_x_pd = memory::primitive_desc(
          { {src_x_tz}, memory::data_type::f32, x->format()},
          mkldnn_engine);
    auto src_y_pd = memory::primitive_desc(
          { {src_y_tz}, memory::data_type::f32, y->format()},
          mkldnn_engine);
    auto src_x_memory = memory(src_x_pd, cast_const_to_void(x_data));
    auto src_y_memory = memory(src_y_pd, cast_const_to_void(y_data));

    srcs_pd.push_back(src_x_pd);
    srcs_pd.push_back(src_y_pd);
    srcs.push_back(src_x_memory);
    srcs.push_back(src_y_memory);

    auto dst_md = memory::desc(
           {dst_tz}, memory::data_type::f32, memory::format::any);

    // create primitive descriptor for sum
    auto sum_pd = sum::primitive_desc(dst_md, scales, srcs_pd);

    // create mkldnn memory for dst
    memory dst_memory = memory(sum_pd.dst_primitive_desc(), z_data);

    std::vector<primitive::at> inputs;
    inputs.push_back(srcs[0]);
    inputs.push_back(srcs[1]);

    // create sum primitive
    auto sum_prim = sum(sum_pd, inputs, dst_memory);

    std::vector<primitive> pipeline;
    pipeline.push_back(sum_prim);
    stream(stream::kind::eager).submit(pipeline).wait();

    z->set_layout(DataLayout::kMKLDNN);
    z->set_format((memory::format)
                  dst_memory.get_primitive_desc().desc().data.format);
  }
};

template <typename DeviceContext, typename T>
class EltwiseAddMKLDNNGradKernel : public framework::OpKernel<T> {
 public:
//  void Compute(const framework::ExecutionContext& ctx) const override {
//    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
//    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
//    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
//
//    const T* dout_data = dout->data<T>();
//    T* dx_data = dx->mutable_data<T>(ctx.GetPlace());
//    T* dy_data = dy->mutable_data<T>(ctx.GetPlace());
//
//    PADDLE_ENFORCE(dout->layout() == DataLayout::kMKLDNN &&
//                   dout->format() != memory::format::format_undef,
//                   "Wrong layout/format set for dout tensor");
//
//    // Just memory copy dout to dx and dy
//    const paddle::platform::Place src_place = ctx.GetPlace();
//    const auto dst_place = src_place;
//    ::paddle::memory::Copy(
//       boost::get<paddle::platform::CPUPlace>(dst_place), dx_data,
//       boost::get<paddle::platform::CPUPlace>(src_place), dout_data,
//       dout->memory_size());
//    ::paddle::memory::Copy(
//       boost::get<paddle::platform::CPUPlace>(dst_place), dy_data,
//       boost::get<paddle::platform::CPUPlace>(src_place), dout_data,
//       dout->memory_size());
//
//    dx->set_layout(DataLayout::kMKLDNN);
//    dx->set_format(dout->format());
//    dy->set_layout(DataLayout::kMKLDNN);
//    dy->set_format(dout->format());
//  }
    void Compute(const framework::ExecutionContext& ctx) const override {
        using Tensor = framework::Tensor;
        auto* x = ctx.Input<Tensor>("X");
        auto* y = ctx.Input<Tensor>("Y");
        auto* out = ctx.Input<Tensor>("Out");
        auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
        auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
        auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
        int axis = ctx.Attr<int>("axis");
//    ElemwiseGradCompute<DeviceContext, T, IdentityGrad<T>, IdentityGrad<T>>(
//        ctx, *x, *y, *out, *dout, axis, dx, dy, IdentityGrad<T>(),
//        IdentityGrad<T>());
     if (platform::is_cpu_place(ctx.GetPlace()) && (x->dims() == y->dims())) {
          auto blas = math::GetBlas<DeviceContext, T>(ctx);

          if (dx) {
              blas.VCOPY(dout->numel(), dout->data<T>(),
                         dx->mutable_data<T>(ctx.GetPlace()));
          }

          if (dy) {
              blas.VCOPY(dout->numel(), dout->data<T>(),
                         dy->mutable_data<T>(ctx.GetPlace()));
          }
      } else {
          ElemwiseGradCompute<DeviceContext, T, IdentityGrad<T>,
                  IdentityGrad<T>>(
                  ctx, *x, *y, *out, *dout, axis, dx, dy, IdentityGrad<T>(),
                  IdentityGrad<T>());
      }

      dx->set_layout(DataLayout::kMKLDNN);
      dx->set_format(dout->format());
      dy->set_layout(DataLayout::kMKLDNN);
      dy->set_format(dout->format());
    }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(elementwise_add, MKLDNN, ::paddle::platform::CPUPlace,
    ops::EltwiseAddMKLDNNKernel<paddle::platform::CPUDeviceContext,
                                float>)
REGISTER_OP_KERNEL(elementwise_add_grad, MKLDNN, ::paddle::platform::CPUPlace,
    ops::EltwiseAddMKLDNNGradKernel<paddle::platform::CPUDeviceContext,
                                    float>)
