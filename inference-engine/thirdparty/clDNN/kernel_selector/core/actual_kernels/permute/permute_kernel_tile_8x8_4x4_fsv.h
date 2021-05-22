﻿// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "permute_kernel_base.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PermuteKernel_tile_8x8_4x4
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class PermuteKernel_tile_8x8_4x4_fsv : public PermuteKernelBase {
public:
    using Parent = PermuteKernelBase;
    using Parent::Parent;
    PermuteKernel_tile_8x8_4x4_fsv() : PermuteKernelBase("permute_tile_8x8_4x4_fsv") {}
    virtual ~PermuteKernel_tile_8x8_4x4_fsv() {}

    bool Validate(const Params& p, const optional_params& o) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const;
    ParamsKey GetSupportedKey() const override;
protected:
    JitConstants GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const;
    CommonDispatchData SetDefault(const permute_params& params) const;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE,
            FusedOpType::SCALE
        };
    }
};
}  // namespace kernel_selector
