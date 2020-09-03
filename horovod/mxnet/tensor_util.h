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

#ifndef HOROVOD_MXNET_TENSOR_UTIL_H
#define HOROVOD_MXNET_TENSOR_UTIL_H

#include <cassert>
#include <mxnet/c_api.h>

#include "../common/common.h"
#include "util.h"

namespace horovod {
namespace mxnet {

using namespace horovod::common;

class TensorUtil {
public:
  static const DataType GetDType(NDArrayHandle tensor);
  static const TensorShape GetShape(NDArrayHandle tensor);
  static const void* GetData(NDArrayHandle tensor);
  static int64_t GetSize(NDArrayHandle tensor);
  static int GetDevice(NDArrayHandle tensor);

  static void Copy(NDArrayHandle output, NDArrayHandle tensor);
  static void DivideTensorInPlace(NDArrayHandle tensor, int value);

private:
  static const size_t kFloat32Size = 4;
  static const size_t kFloat64Size = 8;
  static const size_t kFloat16Size = 2;
  static const size_t kUInt8Size = 1;
  static const size_t kInt32Size = 4;
  static const size_t kInt8Size = 1;
  static const size_t kInt64Size = 8;
};

} // namespace mxnet
} // namespace horovod

#endif // HOROVOD_MXNET_TENSOR_UTIL_H
