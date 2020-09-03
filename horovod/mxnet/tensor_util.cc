// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <nnvm/c_api.h>

#include "tensor_util.h"

namespace horovod {
namespace mxnet {

// Define all types for TensorUtil.
const DataType TensorUtil::GetDType(NDArrayHandle tensor) {
  int dtype;
  CHECK_CALL(MXNDArrayGetDType(tensor, &dtype));

  switch (dtype) {
  case mshadow::kFloat32:
    return DataType::HOROVOD_FLOAT32;
  case mshadow::kFloat64:
    return DataType::HOROVOD_FLOAT64;
  case mshadow::kFloat16:
    return DataType::HOROVOD_FLOAT16;
  case mshadow::kUint8:
    return DataType::HOROVOD_UINT8;
  case mshadow::kInt32:
    return DataType::HOROVOD_INT32;
  case mshadow::kInt8:
    return DataType::HOROVOD_INT8;
  case mshadow::kInt64:
    return DataType::HOROVOD_INT64;
  default:
    throw std::logic_error("GetDType: Type " + std::to_string(dtype) +
                           " is not supported in MPI mode.");
  }
}

// Return shape of tensor (similar to TShape)
const TensorShape TensorUtil::GetShape(NDArrayHandle tensor) {
  TensorShape shape;
  int dim;
  const int64_t* mx_shape;
  CHECK_CALL(MXNDArrayGetShape64(tensor, &dim, &mx_shape));
  for (int idx = 0; idx < dim; idx++) {
    shape.AddDim(mx_shape[idx]);
  }
  return shape;
}

// Return data of tensor
const void* TensorUtil::GetData(NDArrayHandle tensor) {
  void *data;
  CHECK_CALL(MXNDArrayGetData(tensor, &data));
  return data;
}

// Return size of tensor in bytes
int64_t TensorUtil::GetSize(NDArrayHandle tensor) {
  DataType dtype = TensorUtil::GetDType(tensor);
  int64_t element_size = 0;
  switch (dtype) {
  case DataType::HOROVOD_FLOAT32:
    element_size = kFloat32Size;
    break;
  case DataType::HOROVOD_FLOAT64:
    element_size = kFloat64Size;
    break;
  case DataType::HOROVOD_FLOAT16:
    element_size = kFloat16Size;
    break;
  case DataType::HOROVOD_UINT8:
    element_size = kUInt8Size;
    break;
  case DataType::HOROVOD_INT32:
    element_size = kInt32Size;
    break;
  case DataType::HOROVOD_INT8:
    element_size = kInt8Size;
    break;
  case DataType::HOROVOD_INT64:
    element_size = kInt64Size;
    break;
  default:
    throw std::logic_error("Type " + std::to_string(dtype) +
                           " is not supported in MPI mode.");
  }
  return TensorUtil::GetShape(tensor).num_elements() * element_size;
}

// If Tensor on GPU, return device id
// Otherwise return CPU_DEVICE_ID (-1)
int TensorUtil::GetDevice(NDArrayHandle tensor) {
  int dev_type, dev_id;
  CHECK_CALL(MXNDArrayGetContext(tensor, &dev_type, &dev_id));
  if (dev_type == mshadow::gpu::kDevMask)
    return dev_id;
  return CPU_DEVICE_ID;
}

// Copy from tensor to output
void TensorUtil::Copy(NDArrayHandle output, NDArrayHandle tensor) {
  int is_deferred_compute, is_autograd_recording;
  CHECK_CALL(MXNDArraySetIsDeferredCompute(0, &is_deferred_compute));
  CHECK_CALL(MXAutogradSetIsRecording(0, &is_autograd_recording));

  // TODO(leezu) cache handle
  void* ophandle;
  std::string opname = "_copyto";
  CHECK_CALL(NNGetOpHandle(opname.c_str(), &ophandle));

  const int* out_stypes;
  int num_outputs = 1;
  NDArrayHandle* output_ptr = &output;
  CHECK_CALL(MXImperativeInvoke(ophandle, 1, &tensor, &num_outputs, &output_ptr, 0,
                                nullptr, nullptr, &out_stypes));

  CHECK_CALL(MXNDArraySetIsDeferredCompute(is_deferred_compute, &is_deferred_compute));
  CHECK_CALL(MXAutogradSetIsRecording(is_autograd_recording, &is_autograd_recording));
}

// Elementwise division of tensor by value in-place
void TensorUtil::DivideTensorInPlace(NDArrayHandle tensor, int value) {
  throw std::runtime_error("DivideTensorInPlace not yet supported for MXNet.");
}

} // namespace mxnet
} // namespace horovod
