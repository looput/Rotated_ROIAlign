// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include "cuda/vision.h"


// Interface for Python
at::Tensor ROIAlignRotated_forward(const at::Tensor& input,
                            const at::Tensor& rois,
                            const float spatial_scale,
                            const int pooled_height,
                            const int pooled_width,
                            const int sampling_ratio) {
  if (input.type().is_cuda()) {
    return ROIAlignRotated_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
  }
}

at::Tensor ROIAlignRotated_backward(const at::Tensor& grad,
                             const at::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int batch_size,
                             const int channels,
                             const int height,
                             const int width,
                             const int sampling_ratio) {
  if (grad.type().is_cuda()) {
    return ROIAlignRotated_backward_cuda(grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio);
  }
  AT_ERROR("Not implemented on the CPU");
}

