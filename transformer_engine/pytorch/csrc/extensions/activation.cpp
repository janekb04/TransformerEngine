/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "common.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

template <void (*act_func)(const NVTETensor, NVTETensor, cudaStream_t)>
py::object activation_helper(const at::Tensor& input, py::handle quantizer, int shape_divisor = 1) {
  init_extension();
  auto my_quantizer = convert_quantizer(quantizer);
  auto input_tensor = input.contiguous();

  const TensorWrapper& te_input = makeTransformerEngineTensor(input_tensor);
  const auto& te_input_shape = te_input.shape();
  std::vector<size_t> input_shape(te_input_shape.data, te_input_shape.data + te_input_shape.ndim);
  input_shape[input_shape.size() - 1] /= shape_divisor;
  auto fake_tensor_type = input.scalar_type();

  auto [te_output, out] =
      my_quantizer->create_tensor(input_shape, GetTransformerEngineDType(fake_tensor_type));

  // for current scaling, we need to compute activation and amax first and then quantize
  // because we have no amax to rely on as the activation function will change the data range
  // we cannot compute amax on the fly on a block by block basis while quantizing
  // because the amax relies on the entirety of the tensor having undergone the activation
  // amax reduction is not supported here, so no process group needed
  if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    // we compute both the activation and the amax in the activation kernel
    
    // we have to output in higher precision, so we create a tensor for storing that
    auto my_quantizer_none = std::make_unique<NoneQuantizer>(py::none());
    auto [te_output_act, out_act] =
        my_quantizer_none->create_tensor(input_shape, GetTransformerEngineDType(fake_tensor_type));

    // we set the amax of the high precision output (te_output_act) to point to the amax
    // of the quantized output (te_output) which is owned by its quantizer to make sure
    // that it gets populated by act_func
    // usually act_func wouldn't compute the amax for high precision outputs as they don't
    // have the amax set as they don't need it
    auto my_quantizer_cs = static_cast<Float8CurrentScalingQuantizer*>(my_quantizer.get());
    const auto& amax = my_quantizer_cs->amax;
    te_output_act.set_amax(amax.data_ptr(), GetTransformerEngineDType(amax.scalar_type()),
                           getTensorShape(amax));

    // compute the activation in high precision and store it in te_output_act
    // and compute the amax and store it in my_quantizer_cs->amax (ie. te_output's amax)
    NVTE_SCOPED_GIL_RELEASE({
      act_func(te_input.data(), te_output_act.data(), at::cuda::getCurrentCUDAStream());
    });

    // unset the amax for te_output_act, as it is no longer needed
    te_output_act.set_amax(nullptr, DType::kFloat32, te_output_act.defaultShape);

    // verify that the quantizer doesn't require amax reduction
    if (my_quantizer_cs->with_amax_reduction) {
      NVTE_ERROR(
          "per-tensor current scaling amax reduction is not supported in activation functions.");
    }
    
    // set quantization config parameters
    QuantizationConfigWrapper quant_config;
    quant_config.set_force_pow_2_scales(my_quantizer_cs->force_pow_2_scales);
    quant_config.set_amax_epsilon(my_quantizer_cs->amax_epsilon);

    NVTE_SCOPED_GIL_RELEASE({
      // compute te_output's scale from its amax (set above by act_funct through te_output_act)
      nvte_compute_scale_from_amax(te_output.data(), quant_config,
                                   at::cuda::getCurrentCUDAStream());
      // unset the amax for te_output to avoid atomic amax updates in kernel
      te_output.set_amax(nullptr, DType::kFloat32, te_output.defaultShape);
      // quantize high precision activation output to fp8 using te_output's scale
      nvte_quantize_v2(te_output_act.data(), te_output.data(), quant_config,
                       at::cuda::getCurrentCUDAStream());
    });
  } else if (detail::IsFloat8BlockwiseQuantizers(quantizer.ptr())) {
    // sanity check, since activation fusion is not supported for blockwise quantization yet
    // need to raise an error here instead of silently going into act_func with wrong numerics
    NVTE_ERROR("Activation fusion is not supported for blockwise quantization yet.");
  } else {
    NVTE_SCOPED_GIL_RELEASE(
        { act_func(te_input.data(), te_output.data(), at::cuda::getCurrentCUDAStream()); });
  }

  return out;
}

template <void (*act_func)(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t)>
py::object dactivation_helper(const at::Tensor& grad, const at::Tensor& input,
                              py::handle quantizer) {
  init_extension();
  auto my_quantizer = convert_quantizer(quantizer);
  auto input_tensor = input.contiguous();
  auto grad_tensor = grad.contiguous();

  const TensorWrapper& te_input = makeTransformerEngineTensor(input_tensor);
  const TensorWrapper& te_grad = makeTransformerEngineTensor(grad_tensor);
  const auto& te_input_shape = te_input.shape();
  std::vector<size_t> input_shape(te_input_shape.data, te_input_shape.data + te_input_shape.ndim);
  auto fake_tensor_type = input.scalar_type();

  auto [te_output, out] =
      my_quantizer->create_tensor(input_shape, GetTransformerEngineDType(fake_tensor_type));

  NVTE_SCOPED_GIL_RELEASE({
    act_func(te_grad.data(), te_input.data(), te_output.data(), at::cuda::getCurrentCUDAStream());
  });

  return out;
}

py::object gelu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_gelu>(input, quantizer);
}

py::object dgelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dgelu>(grad, input, quantizer);
}

py::object relu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_relu>(input, quantizer);
}

py::object drelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_drelu>(grad, input, quantizer);
}

py::object geglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_geglu>(input, quantizer, 2);
}

py::object qgeglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_qgeglu>(input, quantizer, 2);
}

py::object dgeglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dgeglu>(grad, input, quantizer);
}

py::object dqgeglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dqgeglu>(grad, input, quantizer);
}

py::object reglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_reglu>(input, quantizer, 2);
}

py::object dreglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dreglu>(grad, input, quantizer);
}

py::object swiglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_swiglu>(input, quantizer, 2);
}

py::object dswiglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dswiglu>(grad, input, quantizer);
}

py::object qgelu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_qgelu>(input, quantizer);
}

py::object dqgelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dqgelu>(grad, input, quantizer);
}

py::object srelu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_srelu>(input, quantizer);
}

py::object dsrelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dsrelu>(grad, input, quantizer);
}

}  // namespace transformer_engine::pytorch
