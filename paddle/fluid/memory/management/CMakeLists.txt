cc_library(file_listener SRCS file_listener.cc DEPS glog crypto)
cc_library(gpu_usage_adjustment SRCS gpu_usage_adjustment.cc DEPS glog allocator gpu_info)
cc_library(gpu_resource_management SRCS gpu_resource_management.cc DEPS glog jsoncpp file_listener gpu_usage_adjustment)
