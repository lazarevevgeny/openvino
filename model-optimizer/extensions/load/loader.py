"""
 Copyright (C) 2020 Intel Corporation

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
from collections import defaultdict

from mo.graph.graph import Graph
from mo.utils import class_registration
import telemetry.telemetry as tm


class Loader(object):
    registered_cls = []
    registered_ops = {}
    excluded_replacers = []

    def find_and_replace_pattern(self, graph: Graph):
        self.load(graph)

    def load(self, graph: Graph):
        raise Exception("Define load logic of {} class in its load method".format(
            self.__class__.__name__
        ))

    def run_before(self):
        """
        Returns list of loader classes which this loader must be run before.
        :return: list of classes
        """
        return [LoadFinish]

    def run_after(self):
        """
        Returns list of loader classes which this loader must be run after.
        :return: list of classes
        """
        return []

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.LOADER


class LoadFinish(Loader):
    enabled = True

    def run_before(self):
        return []

    def run_after(self):
        return []

    def load(self, graph: Graph):
        op_cnt = defaultdict(int)
        for node in graph.get_op_nodes():
            op_cnt[node.op] += 1
        sender = tm.Telemetry()
        for op, cnt in op_cnt.items():
            sender.send_event('model_info', 'op_instances', op, cnt)

        sender.send_event('model_info', 'joined_ops_types_used', ','.join(sorted(list(op_cnt.keys()))))

        graph.check_empty_graph('loading from framework')
