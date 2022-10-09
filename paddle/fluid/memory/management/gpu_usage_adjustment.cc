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

#include "paddle/fluid/memory/management/gpu_usage_adjustment.h"

#include <climits>

#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.cc"

namespace paddle {
namespace memory {
namespace management {

bool CUDAAllocatorAdjustor::AdjustMemoryLimit(
    std::shared_ptr<allocation::StatAllocator> allocator, 
    int device_id,
    int64_t new_memory_limit) {
  PADDLE_ENFORCE_NOT_NULL(allocator,
                          platform::errors::NotFound("Invliad parameter"));

  int64_t current_reserved_memory = DEVICE_MEMORY_STAT_CURRENT_VALUE(Reserved, device_id);

  VLOG(0) << "try to adjust device_id " << device_id << " to "
          << string::HumanReadableSize(new_memory_limit) << ", current allocated "
          << string::HumanReadableSize(current_reserved_memory);
  // Resize limit
  platform::ResizeGpuLimitSize(device_id, new_memory_limit);

  if (current_reserved_memory <= new_memory_limit) {
    return true;
  } else {
    auto device_reserved_memory = FreeEmptyMemory(
                          allocator, device_id);
    if (device_reserved_memory <= new_memory_limit) {
      return true;
    } 
  }
  return false;
  // size_t new_limit = new_memory_limit;
  // if (new_memory_limit < current_reserved_memory) {
  //   int64_t free_res = FreeEmptyMemory(allocator, device_id, new_memory_limit);
  //   if (free_res > new_memory_limit) {
      
  //   }
  // }
  // platform::RecordedLimitResize(device_id, new_memory_limit);
  // platform::ResizeGpuLimitSize(device_id, new_memory_limit);
  // if (new_memory_limit >= current_reserved_memory) {
  //   // bool ok =
  //   // paddle::platform::RecordedLimitResize(device_id, new_memory_limit);
  //   bool ok = platform::ResizeGpuLimitSize(device_id, new_memory_limit);;
  //   VLOG(2) << "RecordedLimitResize: " << ok;

  // } else {
  //   // total_region_allocated_bytes_ > new_memory_limit:
  //   // shrink, need to free memory
  //   auto device_reserved_memory = FreeEmptyMemory(allocator, device_id, new_memory_limit);
  //   // auto device_reserved_memory = DEVICE_MEMORY_STAT_CURRENT_VALUE(Reserved, device_id);
  //   VLOG(2) << "after free, alloc size " << device_reserved_memory;
  //   if (device_reserved_memory <= new_memory_limit) {
  //     VLOG(2) << "successful";
  //     return device_reserved_memory;
  //     // device_reserved_memory = new_memory_limit;
  //   } else {
  //     VLOG(2) << "failed";
  //     device_reserved_memory = ;
  //   }
  //   // bool ok = paddle::platform::RecordedLimitResize(device_id, new_limit);
  //   // bool ok = allocator->ResizeLimit(free_res);
  //   bool ok = platform::ResizeGpuLimitSize(device_id, new_memory_limit);
  //   VLOG(2) << "RecordedLimitResize: " << ok;
  // }

  // // platform::RecordedLimitResize(device_id, new_memory_limit);
  // return device_reserved_memory;
}

void CUDAAllocatorAdjustor::GetMemPoolStats(
    std::shared_ptr<detail::BuddyAllocator> allocator, int device_id,
    int64_t& deviceMemPoolSize, int64_t& deviceMemStable) {
  // TODO
  return;
}

int64_t CUDAAllocatorAdjustor::FreeEmptyMemory(
    std::shared_ptr<allocation::StatAllocator> allocator, 
    int device_id) {
  platform::CUDAPlace place(device_id);
  size_t free_res = allocator->Release(place);
  VLOG(2) << "Release: " << free_res;
  return DEVICE_MEMORY_STAT_CURRENT_VALUE(Reserved, device_id);
}

bool GPUUsageAdjustment::ReleaseAllocatorToLimit(int device_id) {
  auto allo = GetGPUAllocator(device_id);
  cuda_allocator_adjustor_->FreeEmptyMemory(allo, device_id);
  bool success = int64_t(platform::RecordedGpuLimitSize(device_id)) > 
        DEVICE_MEMORY_STAT_CURRENT_VALUE(Reserved, device_id);
  return success;
}


bool GPUUsageAdjustment::AdjustMemLimit(int device_id, size_t new_mem_limit) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto cur_info = cur_usage_info_.find(device_id);
  if (cur_info == cur_usage_info_.cend()) {
    auto allo = GetGPUAllocator(device_id);
    if (allo == nullptr) {
      LOG(ERROR) << "Failed to get the allocator of device_id: " << device_id;
      return false;
    }

    GPUUsageInfo usage_info;
    usage_info.gpu_allocator_ = allo;
    usage_info.cur_limit_.mem_limit_ = ULONG_MAX;

    auto ret = cur_usage_info_.emplace(device_id, usage_info);
    if (ret.second == false) {
      return false;
    }

    cur_info = ret.first;
  }

  if (cur_info->second.cur_limit_.mem_limit_ == new_mem_limit) {
    return true;
  }

  VLOG(2) << "Start to manage the mem size limit to " << new_mem_limit
          << "b of device: " << device_id;

  bool success = cuda_allocator_adjustor_->AdjustMemoryLimit(
                             cur_info->second.gpu_allocator_,
                             device_id, 
                             new_mem_limit);
  return success;
}

std::shared_ptr<allocation::StatAllocator> GPUUsageAdjustment::GetGPUAllocator(
    int device_id) {
  // get allocator from allocator_facade
  // down cast to mix
  platform::CUDAPlace place(device_id);
  auto allocator = allocation::AllocatorFacade::Instance().GetAllocator(place);
  PADDLE_ENFORCE_NOT_NULL(
      allocator,
      platform::errors::NotFound("Allocator not found for device %s", place));
  VLOG(0) << "allocator type " << typeid(*allocator.get()).name();
  auto mixed_allocator = std::dynamic_pointer_cast<
                           allocation::MixedMemBestFitAllocator>(
                           allocator);
  auto stat_allocator = std::dynamic_pointer_cast<
                           allocation::StatAllocator>(
                           mixed_allocator->GetDeviceAllocator());
  
  PADDLE_ENFORCE_NOT_NULL(stat_allocator,
                          platform::errors::NotFound(
                            "StatAllocator not found for device %s", place));
  // auto mixed_allocator = stat_allocator->GetDeviceAllocator();

  // auto mixed_allocator =
  //     std::dynamic_pointer_cast<allocation::MixedMemBestFitAllocator>(
  //         stat_allocator->GetDeviceAllocator());
  // PADDLE_ENFORCE_NOT_NULL(mixed_allocator,
  //                         platform::errors::NotFound(
  //                             "MixedAllocator not found for device %s", place));

  // auto gpu_allocator = std::dynamic_pointer_cast<allocation::AutoGrowthBestFitAllocator>(
  //   mixed_allocator->GetDeviceAllocator());
  // PADDLE_ENFORCE_NOT_NULL(
  //     gpu_allocator,
  //     platform::errors::NotFound("MixedDeviceAllocator not found for device %s",
  //                                place));

  return stat_allocator;
}

}  // namespace management
}  // namespace memory
}  // namespace paddle
