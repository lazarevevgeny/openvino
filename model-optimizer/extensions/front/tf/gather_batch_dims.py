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
import logging as log

from extensions.front.SqueezeNormalize import SqueezeNormalize
from extensions.ops.elementwise import Sub
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes
from mo.ops.squeeze import Squeeze
from mo.ops.unsqueeze import Unsqueeze


class GatherWithBatchDimsNormalize(FrontReplacementPattern):
    enabled = True

    def run_before(self):
        return [SqueezeNormalize]

    @staticmethod
    def gather_with_batch_dims(gather: Node):
        graph = gather.graph
        gather_name = gather.soft_get('name', gather.id)
        log.error('The Gather operation "{}" has attribute "batch_dims" not equal to 0 which is supported. '
                  'The transformation inserts Squeeze operation to remove "batch_dims" dimensions. '
                  'It will not work if the dimensions being squeezed are not 1.'.format(gather_name),
                  extra={'is_warning': True})

        batch_dims = gather.batch_dims
        for in_port in range(2):
            gather.in_port(in_port).get_connection().insert_node(
                Squeeze(graph, {'squeeze_dims': int64_array(range(0, batch_dims)),
                                'name': gather_name + '/in_port/' + str(in_port) + '/Squeeze'}).create_node())
        # decrease the axis id by number of batch dimensions
        gather.in_port(2).get_connection().insert_node(create_op_with_const_inputs(graph, Sub,
                                                                                   {1: int64_array(batch_dims)}))
        unsqueeze_node = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array(range(0, batch_dims))})
        gather.out_port(0).get_connection().insert_node(unsqueeze_node)
        gather.batch_dims = 0

        rename_nodes([(gather, gather_name + '/Squeezed'), (unsqueeze_node, gather_name)])

    def find_and_replace_pattern(self, graph: Graph):
        for gather in graph.get_op_nodes(op='Gather'):
            if gather.has_valid('batch_dims') and gather.batch_dims != 0:
                self.gather_with_batch_dims(gather)
