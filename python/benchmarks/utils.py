# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
import time
import torch

# Define a timing function to print some results
def time_tc(iters, prepend, runFun, entry_point, inputs):
    timesCPU = []
    timesCPUAndGPU = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.clock()
        outputs = runFun(entry_point, inputs)
        timesCPU.append(time.clock() - start)
        torch.cuda.synchronize()
        timesCPUAndGPU.append(time.clock() - start)
    print("#################################################################")
    timesCPU = sorted(timesCPU)
    print(
        f"{prepend} Total CPU time to launch kernel: min {int(timesCPU[0] * 1000000.0)}us, p50 {int(timesCPU[int(len(timesCPU) // 2)] * 1000000.0)}us, p90 {int(timesCPU[int(len(timesCPU) * 9 // 10)] * 1000000.0)}us, max {int(timesCPU[len(timesCPU) - 1] * 1000000.0)}us"
    )
    timesCPUAndGPU = sorted(timesCPUAndGPU)
    print(
        f"{prepend} Total CPU launch + GPU kernel time: min {int(timesCPUAndGPU[0] * 1000000.0)}us, p50 {int(timesCPUAndGPU[int(len(timesCPUAndGPU) // 2)] * 1000000.0)}us, p90 {int(timesCPUAndGPU[int(len(timesCPUAndGPU) * 9 // 10)] * 1000000.0)}us, max {int(timesCPUAndGPU[len(timesCPUAndGPU) - 1] * 1000000.0)}us"
    )
