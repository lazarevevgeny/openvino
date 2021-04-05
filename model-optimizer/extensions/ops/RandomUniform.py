# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.ops.op import Op


class RandomUniform(Op):
    op = 'RandomUniform'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset7',

            'infer': self.infer,

            'seed': 0,
            'seed2': 0,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return ['seed', 'seed2']

    @staticmethod
    def infer(node):
        output_shape = node.in_port(0).data.get_value()
        node_name = node.soft_get('name', node.id)
        assert output_shape is not None, 'The output shape for the RandomUniform operation "{}" is not defined' \
                                         ''.format(node_name)
        node.out_port(0).data.set_shape(int64_array(output_shape))
