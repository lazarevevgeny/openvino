"""
 Copyright (C) 2017-2021 Intel Corporation

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

from mo.graph.graph import Node, Graph
from mo.middle.passes.infer import partial_infer
from mo.ops.op import Op


class If(Op):
    """
    If operation is an operation which has an input with condition which defines what sub-graph "then" or "else" to be
    executed.
    """
    op = 'If'

    def __init__(self, graph: Graph, attrs: dict):
        base_attrs = {
            'type': None,
            'op': self.op,
            'then_input_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'else_input_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'then_output_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'else_output_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'back_edges': [],  # a list of dicts with such attrs as from_layer, from_port, etc.
            'then_graph': None,  # an Graph object with a "then" body sub-graph (condition is True)
            'else_graph': None,  # an Graph object with a "else" body sub-graph (condition is Fals)
            'sub_graphs': ['then_graph', 'else_graph'],  # built-in attribute with all sub-graphs
            'infer': self.infer,
            'type_infer': self.type_infer,
        }
        super().__init__(graph, base_attrs, attrs)

    @staticmethod
    def connect_body_input(if_node: Node, condition: bool, if_input_port_idx: int, body_parameter: Node,
                           axis: [int, None] = None, start: [int, None] = None, end: [int, None] = None,
                           stride: [int, None] = None, part_size: [int, None] = None):
        """
        Update the input port map to connect the input port with the specified body parameter

        :param if_node: the If node
        :param condition: the boolean defining a condition (then/else) graph to add connect the body
        :param if_input_port_idx: the input port index to connect
        :param body_parameter: the body parameter node to connect
        :param axis: dimension for input slicing
        :param start: start value of dimension from which to start slicing
        :param end: end value of dimension when to finish slicing
        :param stride: a step value for slicing
        :param part_size: a partial size for slicing, i.e. slicing [start; start + part_size)
        :return: None
        """
        assert if_node.soft_get('op') == 'If'
        assert body_parameter.soft_get('op') == 'Parameter'
        sub_graph = if_node.then_graph if condition else if_node.else_graph
        port_map = if_node.then_input_port_map if condition else if_node.else_input_port_map
        assert body_parameter.id in sub_graph

        port_map.append({'axis': axis, 'stride': stride, 'part_size': part_size, 'start': start, 'end': end,
                         'external_port_id': if_input_port_idx,
                         'internal_layer_id': body_parameter['internal_layer_id']})

    @staticmethod
    def connect_body_output(if_node: Node, condition: bool, if_output_port_idx: int, internal_result: Node,
                            axis: [int, None] = None, start: [int, None] = None, end: [int, None] = None,
                            stride: [int, None] = None, part_size: [int, None] = None):
        """
        Update the output port map to connect the body Result node with the specified output port

        :param if_node: the If node
        :param condition: the boolean defining a condition (then/else) graph to add connect the body
        :param if_output_port_idx: the output port index to connect
        :param internal_result: the body Result node to connect
        :param axis: dimension for output concatenation
        :param start: start value of dimension from which to start concatenation
        :param end: end value of dimension when to finish concatenation
        :param stride: a step value for concatenation
        :param part_size: a partial size for concatenation, i.e. concatenation [start; start + part_size)
        :return: None
        """
        assert if_node.soft_get('op') == 'If'
        assert internal_result.soft_get('op') == 'Result'
        sub_graph = if_node.then_graph if condition else if_node.else_graph
        port_map = if_node.then_output_port_map if condition else if_node.else_output_port_map
        assert internal_result.id in sub_graph

        port_map.append({'axis': axis, 'stride': stride, 'part_size': part_size, 'start': start, 'end': end,
                         'external_port_id': if_output_port_idx,
                         'internal_layer_id': internal_result['internal_layer_id']})

    @staticmethod
    def updated_body_parameters_shape(if_node: Node, condition: bool):
        """
        Update shape for If body parameters.

        :param if_node: The If node
        :param condition: the boolean defining a condition (then/else) graph to add connect the body
        :return: None
        """
        port_map = if_node.then_input_port_map if condition else if_node.else_input_port_map
        for record in port_map:
            body_node = If.get_body_node_by_internal_id(if_node, condition, record['internal_layer_id'])
            # the Parameter may be removed because it was not used in the body, for example, the current iteration
            # number input
            if body_node is not None:
                assert body_node.soft_get('type') == 'Parameter'

                loop_port_idx = record['external_port_id']
                input_shape = if_node.in_port(loop_port_idx).get_connection().get_source().data.get_shape()
                body_node.shape = input_shape.copy()
                log.debug('Updated shape for the body node with internal_id "{}" with value {}'
                          ''.format(record['internal_layer_id'], body_node.shape))

    @staticmethod
    def updated_loop_output_ports_shape_and_value(if_node: Node):
        """
        Update shape and values for Loop output ports. If the number of iterations is dynamic then the corresponding
        dimension for the scan outputs (having "axis" attribute) are set to 1 because MO cannot generate IR with
        undefined dimensions.

        :param if_node: The Loop node
        :return: None
        """
        if_name = if_node.soft_get('name', if_node.id)
        for record in if_node.then_output_port_map:
            body_node = If.get_body_node_by_internal_id(if_node, True, record['internal_layer_id'])
            assert body_node is not None
            assert body_node.soft_get('type') == 'Result'

            loop_port_idx = record['external_port_id']
            output_value = body_node.in_port(0).data.get_value()
            output_shape = body_node.in_port(0).data.get_shape()
            # MO does not support evaluation of Loop scan outputs with const values
            if output_value is not None:
                if_node.out_port(loop_port_idx).data.set_value(output_value)
            else:
                if_node.out_port(loop_port_idx).data.set_shape(output_shape)

    @staticmethod
    def get_body_node_by_internal_id(if_node: Node, condition: bool, internal_id: int):
        sub_graph = if_node.then_graph if condition else if_node.else_graph
        suitable_nodes = sub_graph.get_op_nodes(internal_layer_id=internal_id)
        assert len(suitable_nodes) <= 1, \
            'Expected 0 or 1 node with `internal_layer_id`={}, {} found'.format(internal_id, len(suitable_nodes))
        return suitable_nodes[0] if len(suitable_nodes) == 1 else None

    @staticmethod
    def infer(if_node: Node):
        If.updated_body_parameters_shape(if_node, True)
        If.updated_body_parameters_shape(if_node, False)
        partial_infer(if_node.then_graph)
        partial_infer(if_node.else_graph)
        If.updated_loop_output_ports_shape_and_value(if_node)

    @staticmethod
    def type_infer(if_node: Node):
        from mo.middle.passes.infer import type_infer
        #Loop.update_body_parameters_type(if_node)
        type_infer(if_node.body)
        #Loop.update_loop_output_ports_type(if_node)
