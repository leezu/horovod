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

#ifndef HOROVOD_MXNET_MPI_OPS_H
#define HOROVOD_MXNET_MPI_OPS_H

#include <mxnet/c_api.h>

#include "adapter.h"
#include "tensor_util.h"

namespace horovod {
namespace mxnet {

using namespace horovod::common;

struct MpiOpsParam {
  NDArrayHandle input;
  NDArrayHandle output;
  NDArrayHandle cpu_input;
  NDArrayHandle cpu_output;
  Request::RequestType op_type;
  std::string op_name;
  int root_rank;
  NDArrayHandle splits;
  bool average;
  double prescale_factor;
  double postscale_factor;

  MpiOpsParam(NDArrayHandle input,
              NDArrayHandle output,
              NDArrayHandle cpu_input,
              NDArrayHandle cpu_output,
              const Request::RequestType& op_type, const std::string& op_name,
              int root_rank, bool average,
              NDArrayHandle splits,
              double prescale_factor,
              double postscale_factor)
      : input(input),
        output(output),
        cpu_input(cpu_input),
        cpu_output(cpu_output),
        op_type(op_type),
        op_name(op_name),
        root_rank(root_rank),
        splits(splits),
        average(average),
        prescale_factor(prescale_factor),
        postscale_factor(postscale_factor) {
  }
};

inline MpiOpsParam* CreateMpiOpsParam(NDArrayHandle input,
                                      NDArrayHandle output,
                                      NDArrayHandle cpu_input,
                                      NDArrayHandle cpu_output,
                                      const Request::RequestType& op_type,
                                      const std::string& op_name,
                                      int root_rank, bool average,
                                      NDArrayHandle splits,
                                      double prescale_factor,
                                      double postscale_factor) {
  return new MpiOpsParam(input, output, cpu_input, cpu_output, op_type, op_name,
                         root_rank, average, splits, prescale_factor, postscale_factor);
}

void DeleteMpiOpsParam(void* param) {
  auto ops_param = static_cast<MpiOpsParam*>(param);
  delete ops_param;
}

extern "C" int horovod_mxnet_allreduce_async(NDArrayHandle input,
                                             NDArrayHandle output,
                                             const char* name, bool average,
                                             int priority,
                                             double prescale_factor,
                                             double postscale_factor);
extern "C" int horovod_mxnet_allgather_async(NDArrayHandle input,
                                             NDArrayHandle output,
                                             const char* name, int priority);
extern "C" int horovod_mxnet_broadcast_async(NDArrayHandle input,
                                             NDArrayHandle output,
                                             const char* name, int root_rank,
                                             int priority);
extern "C" int horovod_mxnet_alltoall_async(NDArrayHandle input,
                                            NDArrayHandle output,
                                            const char* name,
                                            NDArrayHandle splits,
                                            int priority);

} // namespace mxnet
} // namespace horovod

#endif // HOROVOD_MXNET_MPI_OPS_H
