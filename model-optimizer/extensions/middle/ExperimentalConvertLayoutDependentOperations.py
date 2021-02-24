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
from extensions.middle.InsertLayoutPropagationTransposes import is_input_data_in_correct_layout
from extensions.middle.pass_separator import PostMiddleStart
from extensions.ops.transpose import Transpose
from mo.graph.graph import Node, Graph
from mo.graph.perm_inputs import get_node_with_permutation
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.op import Op, PermuteAttrs


class ExperimentalConvertLayoutDependentOperations(MiddleReplacementPattern):
    """
    This pass finds all layout dependent operations and insert necessary Transpose operations and updates nodes
    attributes accordingly.
    """
    graph_condition = [lambda graph: graph.graph['cmd_params'].experimental_layout_change]
    enabled = True

    def run_after(self):
        return [PostMiddleStart]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes():
            node_name = node.soft_get('name', node.id)
            if node.has_valid('layout') and node.layout in ['NHWC', 'NDHWC']:
                input = node.in_node()
                output = node.out_node()

                # 2. permute convolution weights first (it uses 'permutation' attribute of the data node)
                permute_input_data_for_node(node)


                # Calculate permutation for further Transpose operations
                permutation = PermuteAttrs.get_nhwc_to_nchw_permutation(len(node.layout))

                # Schematic representation of transformation below
                #
                #                                          \            NCHW                              NCHW
                #            NHWC                       --  \            |  permutation       permutation  |
                #   data-->Convolution(example)-->data  --  /            |      |       NCHW      |        |
                #                                          /   data->Transpose->data->Convolution->data->Transpose->data

                # 1. Insert input Transpose
                #    This Transpose will permute input from original input layout to operation layout
                edge_attrs = graph.get_edge_data(input.id, node.id)[0]
                graph.remove_edge(input.id, node.id)

                input_permute_name = node_name + '/input_transpose'
                input_order_const = Const(graph, {'name': input_permute_name + '/order',
                                                  'value': permutation.perm}).create_node_with_data()
                input_permute_op = Transpose(graph, {'name': input_permute_name})
                input_permute_data_node = input_permute_op.create_node_with_data([input, input_order_const])

                graph.add_edge(input_permute_data_node.id, node.id, **edge_attrs)

                # 2. Insert output Transpose
                #    This Transpose will permute output from operation layout to original input layout
                edge_attrs = graph.get_edge_data(node.id, output.id)[0]
                graph.remove_edge(node.id, output.id)

                input_data_node = Op.create_data_node(graph, node, {'shape': output.shape[permutation.perm]},
                                                      edge_attrs)

                output_permute_name = node_name + '/output_transpose'
                output_order_const = Const(graph, {'name': output_permute_name + '/order',
                                                   'value': permutation.inv}).create_node_with_data()
                Transpose(graph, {'name': output_permute_name}).\
                    create_node_with_data([input_data_node, output_order_const], data_nodes=output)

                # 4. Add permutations for Node
                #    Here we use permutation mechanism where data nodes takes permutation attribute.
                #    And then we call permute_attrs method that permutes node attributes according to permutations on
                #    data nodes.
                node.in_node()['permutation'] = permutation
                node.out_node()['permutation'] = permutation
                node.permute_attrs.permute_attrs(node)

                node.in_node()['permutation'] = None
                node.out_node()['permutation'] = None


def permute_input_data_for_node(node: Node):
    input_permutations = [(in_port, edge_attrs['input_permutation']) for in_port, edge_attrs in
                          node.in_edges().items() if edge_attrs.get('input_permutation') is not None]
    for in_port, input_perm in input_permutations:
        permutation, port_info, func = input_perm
        direction, port = port_info.split(':')
        port = int(port)
        port_to_check = node.in_port(port) if direction == 'input' else node.out_port(port)

        permutation_data_node = get_node_with_permutation(node, port_info)
        # dirty hack to make "permutation" function work correctly
        permutation_data_node['permutation'] = node.in_edges()[in_port]['permutation']
        if not is_input_data_in_correct_layout(node, in_port) and len(port_to_check.data.get_shape()) >= 4:
            permutation(node, port_info, in_port)
    if node.has_and_set('need_shape_inference'):
        node.infer(node)
        node.need_shape_inference = False
