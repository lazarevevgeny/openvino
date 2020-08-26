"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import numpy as np

from extensions.ops.activation_ops import Swish
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph, rename_nodes


class SwishWithSigmoidWithoutBeta(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern with Sigmoid defining the Swish function: Swish(x) = x * Sigmoid(x)
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('sigmoid', dict(op='Sigmoid')),
                ('mul', dict(op='Mul')),
            ],
            edges=[
                ('input', 'sigmoid', {}),
                ('input', 'mul', {}),
                ('sigmoid', 'mul', {}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        sigmoid = match['sigmoid']
        mul = match['mul']

        mul_name = mul.soft_get('name', mul.id)

        swish = Swish(graph, {}).create_node()
        swish.in_port(0).connect(sigmoid.in_port(0).get_source())
        mul.out_port(0).get_connection().set_source(swish.out_port(0))

        rename_nodes([(mul, mul_name + '/TBR'), (swish, mul_name)])


class SwishWithSigmoidWithBeta(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern with Sigmoid defining the Swish function: Swish(x) = x * Sigmoid(x * beta)
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('sigmoid', dict(op='Sigmoid')),
                ('beta', dict()),
                ('mul_beta', dict(op='Mul')),
                ('mul', dict(op='Mul')),
            ],
            edges=[
                ('input', 'mul_beta', {}),
                ('input', 'mul', {}),
                ('beta', 'mul_beta', {}),
                ('mul_beta', 'sigmoid', {}),
                ('sigmoid', 'mul', {}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        beta = match['beta']
        mul = match['mul']
        mul_beta = match['mul_beta']

        # determine the input port of Mul which gets the 'input' node output
        mul_beta_input_port = int(mul_beta.in_port(0).get_connection().get_source().node.id == beta.id)
        mul_name = mul.soft_get('name', mul.id)

        swish = Swish(graph, {}).create_node()
        swish.in_port(0).connect(mul_beta.in_port(mul_beta_input_port).get_source())

        # connect Beta value
        swish.in_port(1).connect(mul_beta.in_port(1 - mul_beta_input_port).get_source())

        mul.out_port(0).get_connection().set_source(swish.out_port(0))

        rename_nodes([(mul, mul_name + '/TBR'), (swish, mul_name)])
