// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

#include "ref_eltwise.hpp"
#include "ref_depthwise.hpp"

using namespace InferenceEngine;

namespace MKLDNNPlugin {

enum LayoutType {
    planar,
    block,
    by_channel
};

struct jit_interpolate_config_params {
    LayoutType layout;
    mkldnn::memory::data_type src_dt;
    mkldnn::memory::data_type dst_dt;
    int src_data_size;
    int dst_data_size;
    int indices_size;
    size_t n, c, d, h, w;
};

struct jit_interpolate_call_args {
    const void *src;
    const int *index;
    void *dst;
    size_t work_amount;
    size_t oc_off;
};

struct jit_uni_interpolate_kernel {
    void (*ker_)(const jit_interpolate_call_args *);

    void operator()(const jit_interpolate_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_interpolate_kernel(jit_interpolate_config_params jcp, const mkldnn_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_interpolate_kernel() {}

    jit_interpolate_config_params jcp_;
    const mkldnn_primitive_attr &attr_;
};

enum class InterpolateMode {
    nearest,
    linear,
    linear_onnx,
    cubic
};

enum class CoordTransMode {
    half_pixel,
    pytorch_half_pixel,
    asymmetric,
    tf_half_pixel_for_nn,
    align_corners
};

enum class NearestMode {
    round_prefer_floor,
    round_prefer_ceil,
    floor,
    ceil,
    simple
};

class MKLDNNInterpolateNode : public MKLDNNNode {
public:
    MKLDNNInterpolateNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNInterpolateNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

private:
    // nn
    void NearestNeighbor_PLN(const float *in_ptr_, float *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW);
    template <typename in_data_t, typename out_data_t>
    void NearestNeighbor_BLK(const in_data_t *in_ptr_, out_data_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW);
    template <typename in_data_t, typename out_data_t>
    void NearestNeighbor_ref(const in_data_t *in_ptr_, out_data_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW);
    // linear
    template <typename in_data_t, typename out_data_t>
    void LinearInterpolation(const in_data_t *in_ptr_, out_data_t *out_ptr_, int B, int C, int ID, int IH, int IW,
                                          float fx, float fy, float fz, int OD, int OH, int OW, int kernel_width, bool antialias);
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false);
    inline void apply_post_ops_scalar(float &dst_value, int index_c);

    inline float coordTransToInput(int outCoord, float scale, int inShape, int outShape);
    inline int nearestRound(float origin, bool isDownsample);

    int blk_size;

    std::vector<int> axes;
    InterpolateMode mode;
    CoordTransMode coordTransMode = CoordTransMode::half_pixel;
    bool antialias = false;
    std::vector<int> padBegin;
    std::vector<int> padEnd;
    NearestMode nearestMode = NearestMode::round_prefer_floor;
    float cubeCoeff = -0.75;

    mkldnn::primitive_attr attr;
    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;

    InferenceEngine::Precision inputPrec, outputPrec;
    size_t srcDataSize, dstDataSize;

    std::shared_ptr<jit_uni_interpolate_kernel> interpolateKernel;
    std::vector<std::shared_ptr<mkldnn::impl::cpu::ref_eltwise_scalar_fwd_t>> eltwise_injectors_ref;
    std::vector<std::shared_ptr<mkldnn::impl::cpu::ref_depthwise_scalar_fwd_t>> depthwise_injectors_ref;
};

}  // namespace MKLDNNPlugin

