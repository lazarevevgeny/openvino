// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class TRANSFORMATIONS_API ConvertNMS4ToLegacy;

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *      Convert NMS-4 directly to legacy NMS because NMS-3 and NMS-1 have different shape infer function
 */


class ngraph::pass::ConvertNMS4ToLegacy: public ngraph::pass::FunctionPass {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;
};

