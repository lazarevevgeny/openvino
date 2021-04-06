// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/random_uniform.hpp"
#include <ngraph/validation_util.hpp>
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/random_uniform.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v7::RandomUniform, "RandomUniform", 7);

op::v7::RandomUniform::RandomUniform(const Output<Node>& arg, int64_t seed, int64_t seed2)
        : Op({arg}), m_seed(seed), m_seed2(seed2)
{
    constructor_validate_and_infer_types();
}

bool op::v7::RandomUniform::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v7_RandomUniform_visit_attributes);
    visitor.on_attribute("seed", m_seed);
    visitor.on_attribute("seed2", m_seed2);
    return true;
}

shared_ptr<Node> op::v7::RandomUniform::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v7_RandomUniform_clone_with_new_inputs);
    return make_shared<op::v7::RandomUniform>(new_args.at(0), m_seed, m_seed2);
}

void op::v7::RandomUniform::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v7_RandomUniform_validate_and_infer_types);

    PartialShape input_shape = get_input_partial_shape(0);
    PartialShape output_shape = PartialShape::dynamic();
    if (input_shape.rank().is_static())
    {
        NGRAPH_CHECK(input_shape.rank() == 1, "The rank of the tensor defining output shape must be equal to 1");
        if (const auto& const_shape = get_constant_from_source(input_value(0)))
        {
            output_shape = PartialShape(const_shape->cast_vector<int64_t>());
        }
    }
    set_input_is_relevant_to_shape(0);

    set_output_type(0, ngraph::element::f32, output_shape);
}

namespace
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;

        runtime::reference::RandomUniform<T>(out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_RandomUniform(const HostTensorPtr& arg, const HostTensorPtr& out)
    {
        bool rc = true;
        int64_t count = 0;
        if (arg->get_element_type() == ngraph::element::i32)
        {
            auto* count_p = arg->get_data_ptr<int32_t>();
            count = count_p[0]; // TODO FIXME must be a product of all elements
        }
        else if (arg->get_element_type() == ngraph::element::i64)
        {
            auto* count_p = arg->get_data_ptr<int64_t>();
            count = count_p[0]; // TODO FIXME must be a product of all elements
        }
        else
        {
            count = 10;
//            NGRAPH_CHECK(
//                    false,
//                    "Unsupported element type for output shape input. Expected int32 or int64.");
        }

        switch (element::f32) // TODO read the type from the op attribute
        {
            NGRAPH_TYPE_CASE(evaluate_RandomUniform, bf16, out, count);
            NGRAPH_TYPE_CASE(evaluate_RandomUniform, f16, out, count);
            NGRAPH_TYPE_CASE(evaluate_RandomUniform, f32, out, count);
            default: rc = false; break;
        }
        return rc;
    }
}

//bool op::v7::RandomUniform::evaluate(const HostTensorVector& outputs,
//                                const HostTensorVector& inputs) const
//{
//    NGRAPH_OP_SCOPE(v7_RandomUniform_evaluate);
//    NGRAPH_CHECK(this,
//                 validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
//    return evaluate_RandomUniform(inputs[0], outputs[0]);
//}
