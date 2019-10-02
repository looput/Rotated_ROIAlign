// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "ROIAlignRotated.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_rotated_forward",&ROIAlignRotated_forward,"ROIAlignRotated_forward");
  m.def("roi_align_rotated_backward",&ROIAlignRotated_backward,"ROIAlignRotated_backward");
}
