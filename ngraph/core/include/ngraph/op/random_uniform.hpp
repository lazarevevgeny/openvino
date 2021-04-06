// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v7
        {
            /// \brief A random numbers tensor generation
            ///
            class NGRAPH_API RandomUniform : public ngraph::op::Op
        {
            public:
            NGRAPH_RTTI_DECLARATION;
            RandomUniform() = default;

            /// \brief Constructs a RandomUniform operation.
            ///
            /// \param output_shape Tensor defining the output tensor shape.
            /// \param seed The seed value.
            /// \param seed2 Another seed value.
            RandomUniform(const Output<Node>& output_shape, int64_t seed=0, int64_t seed2=0);

            bool visit_attributes(AttributeVisitor& visitor) override;
            void validate_and_infer_types() override;

            std::shared_ptr<Node>
            clone_with_new_inputs(const OutputVector& new_args) const override;
//            bool evaluate(const HostTensorVector& outputs,
//                          const HostTensorVector& inputs) const override;
            int64_t get_seed() const;
            int64_t get_seed2() const;

            protected:
            int64_t m_seed;
            int64_t m_seed2;
        };
    }
}
}
