// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// Modifications copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
// =============================================================================

#include <atomic>
#include <exception>
#include <iostream>

#include "../common/operations.h"
#include "util.h"
#include "mpi_ops.h"

namespace horovod {
namespace mxnet {

namespace {

std::atomic_int op_count;

std::string GetOpName(const char* prefix, const char* name) {
  if (name != nullptr) {
    return std::string(prefix) + "." + std::string(name);
  }

  op_count.fetch_add(1);
  return std::string(prefix) + ".noname." + std::to_string(op_count);
}
} // namespace

static const char* ALLREDUCE_OP_TYPE_NAME = "horovod_allreduce";
static const char* ALLGATHER_OP_TYPE_NAME = "horovod_allgather";
static const char* BROADCAST_OP_TYPE_NAME = "horovod_broadcast";
static const char* ALLTOALL_OP_TYPE_NAME = "horovod_alltoall";
static const int MX_CPU_DEVICE_TYPE = 1;
static const int MX_CPU_DEVICE_ID = 0;

inline void InvokeCompleteCallback(EngineCallbackHandle on_complete, const Status& status) {
  if (status.ok()) {
    CHECK_CALL(MXEngineInvokeCallback(on_complete));
  } else {
    CHECK_CALL(MXEngineInvokeCallback(on_complete, status.reason().c_str()));
  }
}

inline const char* GetOpTypeName(Request::RequestType op_type) {
  switch (op_type) {
    case Request::RequestType::ALLREDUCE:
      return ALLREDUCE_OP_TYPE_NAME;
    case Request::RequestType::ALLGATHER:
      return ALLGATHER_OP_TYPE_NAME;
    case Request::RequestType::BROADCAST:
      return BROADCAST_OP_TYPE_NAME;
    case Request::RequestType::ALLTOALL:
      return ALLTOALL_OP_TYPE_NAME;
    default:
      throw std::logic_error("Unsupported Horovod operation type.");
  }
}

bool IsTensorOnCPU(NDArrayHandle tensor) {
  return TensorUtil::GetDevice(tensor) == CPU_DEVICE_ID;
}

void DoHorovodOperation(void*, EngineCallbackHandle on_complete, void* param) {
  ThrowIfError(common::CheckInitialized());

  auto ops_param = static_cast<MpiOpsParam*>(param);
  auto input = ops_param->input;
  auto output = ops_param->output;
  auto name = ops_param->op_name;
  auto average = ops_param->average;
  auto prescale_factor = ops_param->prescale_factor;
  auto postscale_factor = ops_param->postscale_factor;
  auto device = TensorUtil::GetDevice(input);

  auto hvd_tensor = std::make_shared<MXTensor>(input);
  auto hvd_context = std::make_shared<MXOpContext>(device, output);
  std::shared_ptr<Tensor> hvd_output = nullptr;

  Status enqueue_result;
  switch (ops_param->op_type) {
    case Request::RequestType::ALLREDUCE:
      hvd_output = std::make_shared<MXTensor>(output);
      enqueue_result = EnqueueTensorAllreduce(
          hvd_context, hvd_tensor, hvd_output, nullptr, name, device,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      }, (average) ? ReduceOp::AVERAGE : ReduceOp::SUM, prescale_factor, postscale_factor);
      break;
    case Request::RequestType::ALLGATHER:
      enqueue_result = EnqueueTensorAllgather(
          hvd_context, hvd_tensor, nullptr, name, device,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      });
      break;
    case Request::RequestType::BROADCAST:
      if (horovod_rank() != ops_param->root_rank) {
        hvd_output = std::make_shared<MXTensor>(output);
      }

      enqueue_result = EnqueueTensorBroadcast(
          hvd_context, hvd_tensor, hvd_output, ops_param->root_rank,
          nullptr, name, device,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      });
      break;
    case Request::RequestType::ALLTOALL:
    {
      auto hvd_splits = std::make_shared<MXTensor>(ops_param->splits);
      enqueue_result = EnqueueTensorAlltoall(
          hvd_context, hvd_tensor, hvd_splits, nullptr, name, device,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      });
      break;
    }
    default:
      throw std::logic_error("Unsupported Horovod operation type.");
  }

  ThrowIfError(enqueue_result);
}

inline void PushHorovodOperation(Request::RequestType op_type, NDArrayHandle input,
                                 NDArrayHandle output, const char* name,
                                 int priority, int root_rank = -1,
                                 bool average = true,
                                 NDArrayHandle splits = nullptr,
                                 double prescale_factor = 1.0,
                                 double postscale_factor = 1.0) {
  auto op_type_name = GetOpTypeName(op_type);
  auto op_name = GetOpName(op_type_name, name);

  // TODO(leezu) memory leak!
  NDArrayHandle cpu_splits = splits;
  if (splits) {
#if HAVE_CUDA
    // We expect splits to be a tensor on CPU. Create CPU copy if required.
    if(!IsTensorOnCPU(splits)) {
      int ndim, dtype;
      int64_t* mx_shape;
      CHECK_CALL(MXNDArrayGetShape64(splits, &ndim, &mx_shape));
      CHECK_CALL(MXNDArrayGetDType(splits, &dtype));
      CHECK_CALL(MXNDArrayCreate64(mx_shape, ndim, MX_CPU_DEVICE_TYPE, MX_CPU_DEVICE_ID,
                                   1 /* delay_alloc */, dtype, &cpu_splits));
      // Make async copy of input tensor to CPU tensor.
      TensorUtil::Copy(splits, cpu_splits);
    }
#endif
  }
  auto ops_param = CreateMpiOpsParam(input, output,
    nullptr /* cpu_input_tensor */, nullptr /* cpu_output_tensor */,
    op_type, op_name, root_rank, average, cpu_splits, prescale_factor, postscale_factor);

  // Not in-place
  if (input != output) {
    std::vector<NDArrayHandle> inputs {input};
    if (cpu_splits != nullptr) {
      // Add splits tensor to input list to enforce dependency on possible async D2H copy
      inputs.push_back(cpu_splits);
    }
    MXEnginePushAsync(DoHorovodOperation, ops_param, DeleteMpiOpsParam,
                      MX_CPU_DEVICE_TYPE, MX_CPU_DEVICE_ID, inputs.data(), inputs.size(),
                      &output, 1, EngineFnProperty::kCPUPrioritized, priority, op_type_name);
  // In-place
  } else {
    std::vector<NDArrayHandle> inputs {};
    if (cpu_splits) {
      inputs.push_back(cpu_splits);
    }
    MXEnginePushAsync(DoHorovodOperation, ops_param, DeleteMpiOpsParam,
                      MX_CPU_DEVICE_TYPE, MX_CPU_DEVICE_ID, inputs.data(), inputs.size(),
                      &output, 1, EngineFnProperty::kCPUPrioritized, priority, op_type_name);
  }
}

#if HAVE_CUDA
void DoHorovodOperationCudaOnCPU(void*, EngineCallbackHandle on_complete, void* param) {
  ThrowIfError(common::CheckInitialized());

  auto ops_param = static_cast<MpiOpsParam*>(param);
  auto name = ops_param->op_name;
  auto hvd_cpu_buffer = std::make_shared<MXTensor>(ops_param->cpu_input);
  auto hvd_context = std::make_shared<MXOpContext>(
    CPU_DEVICE_ID, ops_param->cpu_output);
  auto average = ops_param->average;
  auto prescale_factor = ops_param->prescale_factor;
  auto postscale_factor = ops_param->postscale_factor;

  Status enqueue_result;
  switch (ops_param->op_type) {
    case Request::RequestType::ALLREDUCE:
      enqueue_result = EnqueueTensorAllreduce(
          hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, nullptr, name, CPU_DEVICE_ID,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      }, (average) ? ReduceOp::AVERAGE : ReduceOp::SUM, prescale_factor, postscale_factor);
      break;
    case Request::RequestType::ALLGATHER:
      enqueue_result = EnqueueTensorAllgather(
          hvd_context, hvd_cpu_buffer, nullptr, name, CPU_DEVICE_ID,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      });
      break;
    case Request::RequestType::BROADCAST:
      enqueue_result = EnqueueTensorBroadcast(
          hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ops_param->root_rank,
          nullptr, name, CPU_DEVICE_ID,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      });
      break;
    case Request::RequestType::ALLTOALL:
    {
      auto hvd_splits = std::make_shared<MXTensor>(ops_param->splits);
      enqueue_result = EnqueueTensorAlltoall(
          hvd_context, hvd_cpu_buffer, hvd_splits, nullptr, name, CPU_DEVICE_ID,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      });
      break;
    }
    default:
      throw std::logic_error("Unsupported Horovod operation type.");
  }

  ThrowIfError(enqueue_result);
}

inline void PushHorovodOperationCudaOnCPU(Request::RequestType op_type, NDArrayHandle input,
                                          NDArrayHandle output, const char* name,
                                          int priority, int root_rank = -1,
                                          bool average = true,
                                          NDArrayHandle splits = nullptr,
                                          double prescale_factor = 1.0,
                                          double postscale_factor = 1.0) {
  auto op_type_name = GetOpTypeName(op_type);
  auto op_name = GetOpName(op_type_name, name);

  NDArrayHandle cpu_input = input;
  NDArrayHandle cpu_output = output;
  // TODO(leezu) memory leak
  if(!IsTensorOnCPU(input)) {
    int ndim, dtype;
    int64_t* mx_shape;
    CHECK_CALL(MXNDArrayGetShape64(input, &ndim, &mx_shape));
    CHECK_CALL(MXNDArrayGetDType(input, &dtype));
    CHECK_CALL(MXNDArrayCreate64(mx_shape, ndim, MX_CPU_DEVICE_TYPE, MX_CPU_DEVICE_ID,
                                 1 /* delay_alloc */, dtype, &cpu_input));
    // Make async copy of input tensor to CPU tensor.
    TensorUtil::Copy(input, cpu_input);
  }
  if(!IsTensorOnCPU(output)) {
    int ndim, dtype;
    int64_t* mx_shape;
    CHECK_CALL(MXNDArrayGetShape64(output, &ndim, &mx_shape));
    CHECK_CALL(MXNDArrayGetDType(output, &dtype));
    CHECK_CALL(MXNDArrayCreate64(mx_shape, ndim, MX_CPU_DEVICE_TYPE, MX_CPU_DEVICE_ID,
                                 1 /* delay_alloc */, dtype, &cpu_output));
    // No need to copy the output tensor; just need to make the handle available
  }

  NDArrayHandle cpu_splits = splits;
  if (splits) {
    // We expect splits to be a tensor on CPU. Create CPU copy if required.
    if(!IsTensorOnCPU(splits)) {
      int ndim, dtype;
      int64_t* mx_shape;
      CHECK_CALL(MXNDArrayGetShape64(splits, &ndim, &mx_shape));
      CHECK_CALL(MXNDArrayGetDType(splits, &dtype));
      CHECK_CALL(MXNDArrayCreate64(mx_shape, ndim, MX_CPU_DEVICE_TYPE, MX_CPU_DEVICE_ID,
                                   1 /* delay_alloc */, dtype, &cpu_splits));
      // Make async copy of input tensor to CPU tensor.
      TensorUtil::Copy(splits, cpu_splits);
    }
  }

  auto ops_param = CreateMpiOpsParam(nullptr, nullptr, output, cpu_input, cpu_output, op_type,
                                     op_name, root_rank, average, cpu_splits, prescale_factor,
                                     postscale_factor);

  if (op_type == Request::RequestType::ALLGATHER ||
      op_type == Request::RequestType::ALLTOALL) {
    // Use out-of-place path for operations that have unknown output size (allgather, alltoall)
    std::vector<NDArrayHandle> inputs {cpu_input};
    if (splits) {
      // Add splits tensor to input list to enforce dependency on possible async D2H copy
      input.push_back(splits);
    }

    MXEnginePushAsync(DoHorovodOperationCudaOnCPU, ops_param, DeleteMpiOpsParam,
                      MX_CPU_DEVICE_TYPE, MX_CPU_DEVICE_ID, inputs.data(), inputs.size(),
                      &cpu_output, 1, EngineFnProperty::kCPUPrioritized, priority, op_type_name);

    // Since cpu_output_tensor is resized in out-of-place path, need
    // to wait for operation to complete before copying to GPU output.
    error_code = MXNDArrayWaitToRead(cpu_output);
    if(error_code) throw std::runtime_error("MXNDArrayWaitToRead failed.");

    // Make async copy of CPU output tensor to output tensor.
    TensorUtil::Copy(cpu_output, output);
  } else {
    // Use in-place otherwise
    MXEnginePushAsync(DoHorovodOperationCudaOnCPU, ops_param, DeleteMpiOpsParam,
                      MX_CPU_DEVICE_TYPE, MX_CPU_DEVICE_ID, nullptr, 0, &cpu_input, 1,
                      EngineFnProperty::kCPUPrioritized, priority, op_type_name);

    // Make async copy of CPU input tensor to output tensor.
    TensorUtil::Copy(cpu_input, output);
  }
}
#endif

extern "C" int horovod_mxnet_allreduce_async(NDArrayHandle input, NDArrayHandle output,
                                             const char* name, bool average,
                                             int priority,
                                             double prescale_factor,
                                             double postscale_factor) {
  try {

#if HAVE_ROCM
  // Averaging left at framework level for ROCm until ScaleBuffer implementation
  // added.
  bool average_in_framework = average;
  average = false;
#endif

#if HAVE_CUDA && !HOROVOD_GPU_ALLREDUCE
  if (IsTensorOnCPU(input) && IsTensorOnCPU(output)) {
    PushHorovodOperation(Request::RequestType::ALLREDUCE, input, output,
                         name, priority, -1, average, nullptr, prescale_factor, postscale_factor);
  } else {
    PushHorovodOperationCudaOnCPU(Request::RequestType::ALLREDUCE, input, output,
                                  name, priority, -1, average, nullptr, prescale_factor, postscale_factor);
  }
#else
  PushHorovodOperation(Request::RequestType::ALLREDUCE, input, output,
                       name, priority, -1, average, nullptr, prescale_factor, postscale_factor);
#endif

#if HAVE_ROCM
  if (average_in_framework) {
    *output /= horovod_size();
  }
#endif

  } catch (const std::exception e) {
    return -1;
  }
  return 0;
}

extern "C" int horovod_mxnet_allgather_async(NDArrayHandle input,
                                             NDArrayHandle output,
                                             const char* name, int priority) {
  try {

#if HAVE_CUDA && !HOROVOD_GPU_ALLGATHER
  if (IsTensorOnCPU(input) && IsTensorOnCPU(output)) {
    PushHorovodOperation(Request::RequestType::ALLGATHER, input, output,
                         name, priority);
  } else {
    PushHorovodOperationCudaOnCPU(Request::RequestType::ALLGATHER, input, output,
                                  name, priority);
  }
#else
  PushHorovodOperation(Request::RequestType::ALLGATHER, input, output,
                       name, priority);
#endif

  } catch (const std::exception e) {
    return -1;
  }
  return 0;
}

extern "C" int horovod_mxnet_broadcast_async(NDArrayHandle input,
                                             NDArrayHandle output,
                                             const char* name, int root_rank,
                                             int priority) {
  try {

#if HAVE_CUDA && !HOROVOD_GPU_BROADCAST
  if (IsTensorOnCPU(input) && IsTensorOnCPU(output)) {
    PushHorovodOperation(Request::RequestType::BROADCAST, input, output,
                         name, priority, root_rank);

  } else {
    PushHorovodOperationCudaOnCPU(Request::RequestType::BROADCAST, input, output,
                                  name, priority, root_rank);
  }
#else
  PushHorovodOperation(Request::RequestType::BROADCAST, input, output,
                       name, priority, root_rank);
#endif

  } catch (const std::exception e) {
    return -1;
  }
  return 0;
}

extern "C" int horovod_mxnet_alltoall_async(NDArrayHandle input,
                                            NDArrayHandle output,
                                            const char* name,
                                            NDArrayHandle splits,
                                            int priority) {
  try {

#if HAVE_CUDA && !HOROVOD_GPU_ALLTOALL
  if (IsTensorOnCPU(input) && IsTensorOnCPU(output)) {
    PushHorovodOperation(Request::RequestType::ALLTOALL, input, output,
                         name, priority, -1, false, splits);

  } else {
    PushHorovodOperationCudaOnCPU(Request::RequestType::ALLTOALL, input, output,
                                  name, priority, -1, false, splits);
  }
#else
  PushHorovodOperation(Request::RequestType::ALLTOALL, input, output,
                       name, priority, -1, false, splits);
#endif

  } catch (const std::exception e) {
    return -1;
  }
  return 0;
}

} // namespace mxnet
} // namespace horovod
