// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <paddle/fluid/platform/device_context.h>

#include <algorithm>
#include <type_traits>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/math/bert_encoder_functor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class SkipLayerNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *X = context.Input<phi::DenseTensor>("X");
    auto *Y = context.Input<phi::DenseTensor>("Y");
    auto *scale = context.Input<phi::DenseTensor>("Scale");
    auto *bias = context.Input<phi::DenseTensor>("Bias");

    auto *X_d = X->data<T>();
    auto *Y_d = Y->data<T>();
    auto *scale_d = scale->data<T>();
    auto *bias_d = bias->data<T>();
    float epsilon = context.Attr<float>("epsilon");
    int begin_norm_axis = context.Attr<int>("begin_norm_axis");

    auto *out = context.Output<phi::DenseTensor>("Out");
    out->Resize(X->dims());
    auto &dev_ctx = context.template device_context<phi::GPUContext>();
    auto *output_d = dev_ctx.Alloc<T>(out, out->numel() * sizeof(T));

    size_t num = 1;
    for (size_t i = 0; i < X->dims().size(); i++) {
      num *= X->dims()[i];
    }
    int hidden = X->dims()[2];
    auto &device_ctx = context.template device_context<DeviceContext>();
    operators::math::SkipLayerNormFunctor<T> skip_layer_norm_func;

    if (std::is_same<T, paddle::platform::float16>::value) {
      const half *X_new = reinterpret_cast<const half *>(X_d);
      const half *Y_new = reinterpret_cast<const half *>(Y_d);
      const half *scale_new = reinterpret_cast<const half *>(scale_d);
      const half *bias_new = reinterpret_cast<const half *>(bias_d);
      half *output_new = reinterpret_cast<half *>(output_d);
      operators::math::SkipLayerNormFunctor<half> skip_layer_norm_func;
      skip_layer_norm_func(num,
                           hidden,
                           X_new,
                           Y_new,
                           scale_new,
                           bias_new,
                           output_new,
                           epsilon,
                           device_ctx.stream());
    } else {
      operators::math::SkipLayerNormFunctor<T> skip_layer_norm_func;
      skip_layer_norm_func(num,
                           hidden,
                           X_d,
                           Y_d,
                           scale_d,
                           bias_d,
                           output_d,
                           epsilon,
                           device_ctx.stream());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 10000
PD_REGISTER_STRUCT_KERNEL(skip_layernorm,
                          GPU,
                          ALL_LAYOUT,
                          ops::SkipLayerNormKernel,
                          float,
                          plat::float16) {}
#else
PD_REGISTER_STRUCT_KERNEL(
    skip_layernorm, GPU, ALL_LAYOUT, ops::SkipLayerNormKernel, float) {}
#endif
