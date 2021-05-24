﻿// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class resample_kernel_selector : public kernel_selector_base {
public:
    static resample_kernel_selector& Instance() {
        static resample_kernel_selector instance_;
        return instance_;
    }

    resample_kernel_selector();

    virtual ~resample_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params, const optional_params& options) const override;
};
}  // namespace kernel_selector
