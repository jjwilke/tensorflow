/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/backends/profiler/gpu/nvtx_utils.h"

#include "third_party/gpus/cuda/include/nvtx3/nvToolsExt.h"
#include "tensorflow/tsl/platform/platform.h"

namespace xla {
namespace profiler {

/*static*/ std::stack<std::string> &NVTXRangeTracker::GetRangeStack() {
  static thread_local std::stack<std::string> range_stack;
  return range_stack;
}

NvtxContext::NvtxContext(const std::string& name)
{
  NvtxEnter(name);
}

NvtxContext::~NvtxContext()
{
  NvtxExit();
}

void NvtxEnter(const std::string& name){
  nvtxRangePush(name.c_str());
}

void NvtxExit(){
  nvtxRangePop();
}

}  // namespace profiler
}  // namespace xla
