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

from extensions.ops.upsample import UpsampleOp
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr, get_onnx_opset_version
from mo.graph.graph import Node
from mo.utils.error import Error


class ResizeExtractor(FrontExtractorOp):
    op = 'Resize'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        onnx_opset_version = get_onnx_opset_version(node)
        if onnx_opset_version is not None and onnx_opset_version >= 11:
            raise Error("ONNX Resize operation from opset {} is not supported.".format(onnx_opset_version))
        mode = onnx_attr(node, 'mode', 's', default=b'nearest').decode()
        UpsampleOp.update_node_stat(node, {'mode': mode})
        return cls.enabled
