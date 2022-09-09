// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/mixed_mem_best_fit_allocator.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

// PADDLE_DEFINE_EXPORTED_bool(
//     init_allocated_mem, false,
//     "It is a mistake that the values of the memory allocated by "
//     "BuddyAllocator are always zeroed in some op's implementation. "
//     "To find this error in time, we use init_allocated_mem to indicate "
//     "that initializing the allocated memory with a small value "
//     "during unit testing.");

namespace paddle {
namespace memory {
namespace allocation {

phi::Allocation* MixedMemBestFitAllocator::AllocateImpl(size_t size) {
  PADDLE_ENFORCE_NOT_NULL(
      underlying_device_allocator_,
      platform::errors::InvalidArgument("Underlying device allocator of "
                                        "MixedMemBestFitAllocator is nullptr"));

  PADDLE_ENFORCE_NOT_NULL(
      underlying_host_allocator_,
      platform::errors::InvalidArgument("Underlying host allocator of "
                                        "MixedMemBestFitAllocator is nullptr"));

  std::lock_guard<std::mutex> lock(mtx_);

  // try allocate gpu memory
  try {
    phi::Allocator::AllocationPtr device_allocation = 
                    underlying_device_allocator_->Allocate(size);
    // VLOG(0) << "Allocate GPU memory";
    if (device_allocation != nullptr) {
      // VLOG(0) << "GPU allocate success";
      return device_allocation.release();
    }
  } catch (BadAlloc &ex) {
    // VLOG(0) << "Allocate GPU memory fail. Try to allocate CUDAPinnedMemory";
  }
  
  // try allocate CUDAPinnedMemory
  // VLOG(0) << "Start allocate CUDAPinnedMemory";
  try {
    phi::Allocator::AllocationPtr host_allocation = 
                  underlying_host_allocator_->Allocate(size);
    if (host_allocation != nullptr) {
      VLOG(0) << "CUDA Pinned Memory";
      void* dev_ptr;
      
      void* host_ptr = host_allocation->ptr();
      // VLOG(0) << "1";
      PADDLE_ENFORCE_GPU_SUCCESS(cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0));
      PADDLE_ENFORCE_NOT_NULL(dev_ptr, platform::errors::ResourceExhausted(
                                          "cudaHostGetDevicePointer failed")); 
      // VLOG(0) << "2";
      auto aligned_size = host_allocation->size();
      devptr2allocation_.insert({dev_ptr, host_allocation.release()});
      phi::Allocation* tmp_alloc = new Allocation(dev_ptr, aligned_size, device_place_);
      // devptr2hostptr_.insert({dev_ptr, {host_ptr, size}});
      
      return tmp_alloc;
    }
    // void* host_ptr = underlying_host_allocator_->Alloc(size);
    // PADDLE_ENFORCE_NOT_NULL(host_ptr, platform::errors::ResourceExhausted(
    //                                       "cudaHostAlloc failed"));
    // // VLOG(0) << 1;
    // void* dev_ptr;
    // PADDLE_ENFORCE_GPU_SUCCESS(cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0));
    // PADDLE_ENFORCE_NOT_NULL(dev_ptr, platform::errors::ResourceExhausted(
    //                                       "cudaHostGetDevicePointer failed"));
    // VLOG(10) << "system allocator converted host_ptr " << host_ptr
    //           << " to dev_ptr: " << dev_ptr << ", size: " << size;

    // // VLOG(0) << 2;
    
    // Allocation* tmp_alloc = new Allocation(dev_ptr, size, device_place_);
    // VLOG(0) << "Allocate CUDAPinnedMemory success!";
    // return tmp_alloc;

  } catch (...) {
    VLOG(0) << "Allocate CUDAPinnedMemory error in MixedMemBestFitAllocator!";
    throw;
  }
  
  return nullptr;
}

void MixedMemBestFitAllocator::FreeImpl(phi::Allocation* allocation) {
  const auto place = allocation->place();
  std::lock_guard<std::mutex> lock(mtx_);
  // if (place.GetType() == phi::AllocationType::GPU) {
  //   underlying_device_allocator_->Free(allocation);
  // } else {
  //   underlying_host_allocator_->Free(allocation);
  // }
  
  auto it = devptr2allocation_.find(allocation->ptr());
  if (it == devptr2allocation_.end()) {
    underlying_device_allocator_->Free(allocation);
    // VLOG(0) << "free gpu";
    // platform::MemEvenRecorder::Instance().PopMemRecord(
    //     static_cast<void*>(allocation), place);
  } else {
    // VLOG(0) << "before free";
    underlying_host_allocator_->Free(it->second);
    // VLOG(0) << "before erase";
    devptr2allocation_.erase(it);
    // VLOG(0) << "before return";
    delete allocation;
  }

  VLOG(9) << "FreeImpl called, place: " << place
          << ", addr: " << allocation->ptr()
          << ", size: " << allocation->size();
  return;
}

uint64_t MixedMemBestFitAllocator::ReleaseImpl(const platform::Place& place) {
  VLOG(9) << "ReleaseImpl called, place: " << place;
  underlying_host_allocator_->Release(cpu_place_);

  return underlying_device_allocator_->Release(place);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
