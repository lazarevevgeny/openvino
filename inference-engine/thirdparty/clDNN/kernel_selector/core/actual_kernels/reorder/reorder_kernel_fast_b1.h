﻿// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reorder_kernel_base.h"

namespace kernel_selector {
class ReorderKernelFastBatch1 : public ReorderKernelBase {
public:
    ReorderKernelFastBatch1() : ReorderKernelBase("reorder_data_fast_b1") {}

    bool Validate(const Params& p, const optional_params& o) const override;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    JitConstants GetJitConstants(const reorder_params& params) const override;
    DispatchData SetDefault(const reorder_params& arg) const override;
};
}  // namespace kernel_selector