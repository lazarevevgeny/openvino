//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
//*****************************************************************************

#include "resize.hpp"
#include <map>
#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                static const std::unordered_set<std::string> supported_modes = {
                    "nearest", "linear", "cubic"};

                static const std::unordered_set<std::string> supported_transforms = {
                    "half_pixel",
                    "pytorch_half_pixel",
                    "align_corners",
                    "asymmetric",
                    "tf_half_pixel_for_nn"};

                using InterpolateMode = ngraph::op::v4::Interpolate::InterpolateMode;
                static const std::map<std::string, InterpolateMode> interp_mode_map = {
                    {"nearest", InterpolateMode::nearest},
                    {"linear", InterpolateMode::linear_onnx},
                    {"cubic", InterpolateMode::cubic}};

                using Transform_mode = ngraph::op::v4::Interpolate::CoordinateTransformMode;
                static const std::map<std::string, Transform_mode> transform_mode_map = {
                    {"half_pixel", Transform_mode::half_pixel},
                    {"pytorch_half_pixel", Transform_mode::pytorch_half_pixel},
                    {"align_corners", Transform_mode::align_corners},
                    {"asymmetric", Transform_mode::asymmetric},
                    {"tf_half_pixel_for_nn", Transform_mode::tf_half_pixel_for_nn}};

                using Nearest_mode = ngraph::op::v4::Interpolate::NearestMode;
                static const std::map<std::string, Nearest_mode> nearest_mode_map = {
                    {"round_prefer_floor", Nearest_mode::round_prefer_floor},
                    {"round_prefer_ceil", Nearest_mode::round_prefer_ceil},
                    {"floor", Nearest_mode::floor},
                    {"ceil", Nearest_mode::ceil},
                    {"simple", Nearest_mode::simple}};

                static bool is_supported_str_value(const std::unordered_set<std::string>& modes,
                                                   const std::string& checked_mode)
                {
                    return std::find(modes.begin(), modes.end(), checked_mode) != modes.end();
                }

                template <typename T>
                static T convert(const std::map<std::string, T>& converting_map,
                                 const std::string& mode)
                {
                    T result = T{};
                    auto it = converting_map.find(mode);
                    if (it != converting_map.end())
                    {
                        result = it->second;
                    }
                    return result;
                }

                using InterpolateV4Attrs = ngraph::op::v4::Interpolate::InterpolateAttrs;

                InterpolateV4Attrs get_resize_attrs(const onnx_import::Node& node)
                {
                    auto get_str_attr = [&node](const std::string& name,
                                                const std::string& default_value) {
                        return node.get_attribute_value<std::string>(name, default_value);
                    };

                    const auto mode = get_str_attr("mode", "nearest");
                    const auto transform_mode =
                        get_str_attr("coordinate_transformation_mode", "half_pixel");
                    const auto nearest_mode = get_str_attr("nearest_mode", "round_prefer_floor");

                    bool is_mode_supported = is_supported_str_value(supported_modes, mode);
                    if (!is_mode_supported)
                    {
                        std::string supported_modes_str = "";
                        for (const auto& mode_name : supported_modes)
                        {
                            supported_modes_str += (mode_name + ", ");
                        }
                        CHECK_VALID_NODE(node,
                                         is_mode_supported,
                                         mode,
                                         " - this type of interpolation mode is not supported."
                                         " Choose one of the following modes: ",
                                         supported_modes_str);
                    }

                    bool is_transform_mode_supported =
                        is_supported_str_value(supported_transforms, transform_mode);
                    if (!is_transform_mode_supported)
                    {
                        std::string supported_modes_str = "";
                        for (const auto& mode_name : supported_transforms)
                        {
                            supported_modes_str += (mode_name + ", ");
                        }
                        CHECK_VALID_NODE(node,
                                         is_transform_mode_supported,
                                         transform_mode,
                                         " - this type of coordinate transformation mode is not "
                                         "supported. Choose one of the following modes: ",
                                         supported_modes_str);
                    }

                    InterpolateV4Attrs attrs;
                    attrs.mode = convert(interp_mode_map, mode);
                    attrs.coordinate_transformation_mode =
                        convert(transform_mode_map, transform_mode);
                    attrs.nearest_mode = convert(nearest_mode_map, nearest_mode);
                    attrs.antialias = false;
                    attrs.cube_coeff = node.get_attribute_value<float>("cubic_coeff_a", -0.75);

                    auto zero_pad = std::vector<size_t>(1, 0);

                    attrs.pads_begin = zero_pad;
                    attrs.pads_end = zero_pad;

                    return attrs;
                }

                std::shared_ptr<ngraph::Node>
                    calculate_output_shape_based_on_scales(const Output<ngraph::Node>& data,
                                                           const Output<ngraph::Node>& scales)
                {
                    const auto& data_shape = data.get_partial_shape();
                    const auto& scales_shape = scales.get_partial_shape();

                    if (ngraph::op::is_constant(scales.get_node()) && data_shape.is_static())
                    {
                        const auto scales_const =
                            as_type_ptr<default_opset::Constant>(scales.get_node_shared_ptr());

                        const auto scales_vector = scales_const->cast_vector<float>();
                        const auto data_static_shape = data_shape.to_shape();

                        std::vector<int64_t> output_shape;
                        for (size_t i = 0; i < data_static_shape.size(); ++i)
                        {
                            output_shape.push_back(
                                std::floor(data_static_shape.at(i) * scales_vector.at(i)));
                        }
                        auto output_shape_const = default_opset::Constant::create(
                            element::u64, Shape({output_shape.size()}), output_shape);

                        return output_shape_const;
                    }

                    const auto shape_of_data = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::ShapeOf>(data), scales.get_element_type());
                    const auto multiply =
                        std::make_shared<default_opset::Multiply>(shape_of_data, scales);
                    const auto output_shape =
                        std::make_shared<default_opset::Convert>(multiply, ngraph::element::i64);

                    return output_shape;
                }

                OutputVector build_resize(const Node& node,
                                          const std::shared_ptr<ngraph::Node>& output_shape,
                                          const AxisSet& axes)
                {
                    auto attrs = get_resize_attrs(node);

                    const auto inputs = node.get_ng_inputs();
                    const auto& data = inputs.at(0);

                    std::vector<int64_t> axes_vector;
                    axes_vector.insert(axes_vector.end(), axes.begin(), axes.end());

                    auto axes_const = default_opset::Constant::create(
                        ngraph::element::i64, Shape({axes_vector.size()}), axes_vector);
                    return {std::make_shared<ngraph::op::v4::Interpolate>(
                        data, output_shape, axes_const, attrs)};
                }
            }

            namespace set_11
            {
                OutputVector resize(const onnx_import::Node& node)
                {
                    // roi input (inputs.at(2)) is ignored because it is used only
                    // in "tf_crop_and_resize" which is not handled now
                    const auto inputs = node.get_ng_inputs();
                    const auto& data = inputs.at(0);
                    const auto& data_shape = data.get_partial_shape();

                    if (inputs.size() == 4) // sizes input is provided
                    {
                        const auto& sizes = inputs.at(3);
                        const auto& sizes_shape = sizes.get_partial_shape();

                        CHECK_VALID_NODE(
                            node,
                            (sizes_shape.is_static() || data_shape.rank().is_static()),
                            " Data rank or shape of sizes input is required to be static.");

                        size_t axes_size = sizes_shape.is_static() ? sizes_shape[0].get_length()
                                                                   : data_shape.rank().get_length();

                        return build_resize(node,
                                            sizes.get_node_shared_ptr(),
                                            AxisSet(common::get_monotonic_range(axes_size)));
                    }

                    const auto& scales = inputs.at(2);
                    const auto& scales_shape = scales.get_partial_shape();

                    CHECK_VALID_NODE(
                        node,
                        (scales_shape.is_static() || data_shape.rank().is_static()),
                        " Data rank or shape of scales input is required to be static.");

                    size_t axes_size = scales_shape.is_static() ? scales_shape[0].get_length()
                                                                : data_shape.rank().get_length();

                    const auto output_shape = calculate_output_shape_based_on_scales(data, scales);

                    return build_resize(
                        node, output_shape, AxisSet(common::get_monotonic_range(axes_size)));
                }
            } // namespace set_11

            namespace set_1
            {
                OutputVector resize(const onnx_import::Node& node)
                {
                    const auto inputs = node.get_ng_inputs();
                    const auto& data = inputs.at(0);
                    const auto& scales = inputs.at(1);

                    const auto& data_shape = data.get_partial_shape();
                    const auto& scales_shape = scales.get_partial_shape();

                    auto attrs = get_resize_attrs(node);

                    CHECK_VALID_NODE(
                        node,
                        (scales_shape.is_static() || data_shape.rank().is_static()),
                        " Data rank or shape of Scales input is required to be static.");

                    size_t axes_size = scales_shape.is_static() ? scales_shape.to_shape().at(0)
                                                                : data_shape.rank().get_length();
                    std::vector<int64_t> axes;
                    for (int ax = 0; ax < axes_size; ++ax)
                    {
                        axes.push_back(ax);
                    }

                    auto axes_const = default_opset::Constant::create(
                        ngraph::element::i64, Shape({axes.size()}), axes);
                    return {std::make_shared<ngraph::op::v4::Interpolate>(
                        data, scales, axes_const, attrs)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph