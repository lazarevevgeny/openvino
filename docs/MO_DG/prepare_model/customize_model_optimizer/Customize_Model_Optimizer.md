# Model Optimizer Customization {#openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer}
Model Optimizer extensibility mechanism allows to support new operations and custom transformations to generate
optimized IR. This mechanism is a core part of the Model Optimizer and whole Model Optimizer is developed using it,
so the Model Optimizer itself is a huge set of examples on how to add custom logic to support your model.

There are several cases when the customization is needed:

* The model contains operation(s) not known for the Model Optimizer, but these operation(s) could be expressed as a 
combination of supported operations. In this case a custom transformation should be implemented to replace unsupported
operation(s) with supported ones.
* The model contains sub-graph of operations which can be replaced with a smaller number of operations to get the better
performance. This example corresponds to so called fusing transformations. For example, replace a sub-graph performing
the following calculation $x / (1.0 + e^{-(beta * x)})$ to a single operation of type Swish.
* The model contains custom framework operation (the operation which is not a part of official operation set of the
framework) which was developed using the framework extensibility mechanism. In this case the Model Optimizer should know
how to treat the operation and generate an IR for it.

It is necessary to figure out how the Model Optimizer represents a model in memory and convert it to IR before going
into details of the Model Optimizer extensibility mechanism.

> **NOTE**: All paths in this document are provided relatively to the Model Optimizer installation directory if not
stated otherwise.

## Model Representation in Memory
The model can be represented as a directed graph where nodes are operations and edges correspond to data passing from a
producer operation (node) to a consumer operation (node).

Model Optimizer uses Python class `mo.graph.graph.Graph` instance to represent the computation graph in memory during
the model conversion. This class is inherited from `networkx.MultiDiGraph` class of the standard `networkx` Python
library and provides many convenient methods to traverse and modify the graph. Refer to the `mo/graph/graph.py` file for
the examples.

Model Optimizer keeps all necessary information about the operation in the node attributes. Model Optimizer uses class
`mo.graph.graph.Node` defined in the  `mo/graph/graph.py` file which is a wrapper on top of a `networkx` node attributes
dictionary and provides many convenient methods to work with the nodes. In particular, the node `my_node` attribute with
name `my_attr` can be obtained from the node with the following code `my_node.my_attr` which is equivalent to obtaining
attribute with name  `'my_attr'` in the `graph.node['my_node']` dictionary. Refer to the `mo/graph/graph.py` for the
class implementation details.

An operation may have several inputs and outputs. For example, operation [Split](../../../ops/movement/Split_1.md) has
two inputs: data to split and axis to split along, and variable number of outputs depending on attribute `num_splits`.
Each input data to the operation is passed to a specific operation **input port**. Operation produces output data from
the **output port**. Ports are numbered from 0 for input and output independently. Model Optimizer has classes
`mo.graph.port.Port` and `mo.graph.connection.Connection` which are useful abstraction to perform graph modifications
like nodes connecting/re-connecting and graph traversing. These classes are widely used in the Model Optimizer code so
it is easy to find a lot of usage examples.

There is no dedicated class corresponding to an edge, so low-level graph manipulation is needed to get access to
edge attributes if needed. Meanwhile most manipulations with nodes connections should be done with help of
`mo.graph.connection.Connection` and `mo.graph.port.Port` classes. Thus, low-level graph manipulation is strongly not
recommended.

Further details and examples related to a model representation in memory are given in the sections below and provided in
the context for a better explanation.

## Model Conversion Pipeline
The model conversion pipeline can be represented with the following diagram:

![Model Conversion pipeline](../../../img/MO_conversion_pipeline.svg)

Lets review each conversion step in details.

### Model Loading
Model Optimizer gets as input a trained model file. The model loader component of the Model Optimizer reads the model
file using Python bindings provided with the framework and builds in-memory representation of the computation graph.
There is a separate loader for each supported framework. These loaders are implemented in the `extensions/load/<FRAMEWORK>/loader.py`
files of the Model Optimizer.

The result of the model loading step is a `Graph` object which can be depicted like in the following example:

![Graph After Load](../../../img/MO_graph_after_loader.svg)

Model Optimizer loader saves a operation instance framework description (usually it is a Protobuf message) into a node
attribute usually named `pb`. It is important that this is a **framework-specific** description of the operation.
That means that the same operation Convolution which performs the same calculations may be represented differently in
various frameworks.

In the example above the Operation 2 has one input and two outputs. The tensor produced from the output port 0 is
consumed with the Operation 5 (input port 0) and Operation 3 (input port 1). The tensor produced from the output port 1
is consumed with the Operation 4 (input port 0).

Each edge has two attributes `in` and `out` containing input port number of the consumer node and output port
number of the producer node. These attribute describes the fact that nodes are operations getting some inputs and
producing some outputs. But the nodes themselves are "black boxes" from the Model Optimizer perspective because they
don't contain required information about operations they perform.

### Operations Attributes Extracting
The next step is to parse framework-dependent operation representation saved in a node attribute and update the node
attributes with operation specific attributes. There are two ways to do this.

1.  The extractor extension approach. This is recommended way to extract attributes for the operation and it is
explained in details in the section about extensions below.

2.  The legacy approach with built-in extractor. The file `mo/front/<FRAMEWORK>/extractor.py` (for example, the one for
Caffe) has a dictionary with extractors for specific operation types. The key in the dictionary is the type of the
operation to trigger the extractor for and the value is the function to perform attributes extracting. The function has
one parameter -- node to extract the attributes from. This is a legacy and non-extensible approach so should be avoided.
It will be removed in the future versions of the Model Optimizer.

The result of the operations attributes extracting step can be depicted like in the following example:

![Graph After Attributes Extraction](../../../img/MO_graph_after_extractors.svg)

The only difference in the graph from the previous step is that nodes contain dictionary with extracted attributes and
some operation-specific attributes needed for the Model Optimizer. But starting from this step the Model Optimizer does
not need the original representation of the operations/model and uses just Model Optimizer representation (there are
some very specific cases when the Model Optimizer still uses the `pb` attribute and they are slightly covered in this
document). Detailed list of common attributes and their meaning is provided below in the section corresponding to the
Model Optimizer operations.

### Front Phase Transformations

### Partial Inference

### Middle Phase Transformations

### NHWC to NCHW Layout Change

### Back Phase Transformations

### Intermediate Representation Emitting




The detailed solutions for the examples above are given later, the next subsection shows what is common in all three examples.


Model Optimizer searches for each layer of the input model in the list of known layers before building the model's internal representation, optimizing the model, and producing the Intermediate Representation.

The list of known layers is different for each of supported frameworks. To see the layers supported by your framework, refer to the [corresponding section](../Supported_Frameworks_Layers.md).

Custom layers are layers that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

## Caffe\* Models with Custom Layers <a name="caffe-models-with-custom-layers"></a>

You have two options if your Caffe\* model has custom layers:

*   **Register the custom layers as extensions to the Model Optimizer**. For instructions, see [Extending Model Optimizer with New Primitives](Extending_Model_Optimizer_with_New_Primitives.md). When your custom layers are registered as extensions, the Model Optimizer generates a valid and optimized Intermediate Representation. You only need to write a small chunk of Python\* code that lets the Model Optimizer:

    *   Generate a valid Intermediate Representation according to the rules you specified
    *   Be independent from the availability of Caffe on your computer
	
*   **Register the custom layers as Custom and use the system Caffe to calculate the output shape of each Custom Layer**, which is required by the Intermediate Representation format. For this method, the Model Optimizer requires the Caffe Python interface on your system. When registering the custom layer in the `CustomLayersMapping.xml` file, you can specify if layer parameters should appear in Intermediate Representation or if they should be skipped. To read more about the expected format and general structure of this file, see [Legacy Mode for Caffe* Custom Layers](Legacy_Mode_for_Caffe_Custom_Layers.md). This approach has several limitations:

    *   If your layer output shape depends on dynamic parameters, input data or previous layers parameters, calculation of output shape of the layer via Caffe can be incorrect. In this case, you need to patch Caffe on your own.
	
    *   If the calculation of output shape of the layer via Caffe fails inside the framework, Model Optimizer is unable to produce any correct Intermediate Representation and you also need to investigate the issue in the implementation of layers in the Caffe and patch it.
	
    *   You are not able to produce Intermediate Representation on any machine that does not have Caffe installed. If you want to use Model Optimizer on multiple machines, your topology contains Custom Layers and you use `CustomLayersMapping.xml` to fallback on Caffe, you need to configure Caffe on each new machine. 
	
	For these reasons, it is best to use the Model Optimizer extensions for Custom Layers: you do not depend on the framework and fully control the workflow.

If your model contains Custom Layers, it is important to understand the internal workflow of Model Optimizer. Consider the following example.

**Example**:

The network has:

*   One input layer (#1)
*   One output Layer (#5)
*   Three internal layers (#2, 3, 4)

The custom and standard layer types are:

*   Layers #2 and #5 are implemented as Model Optimizer extensions.
*   Layers #1 and #4 are supported in Model Optimizer out-of-the box.
*   Layer #3 is neither in the list of supported layers nor in extensions, but is specified in CustomLayersMapping.xml.

> **NOTE**: If any of the layers are not in one of three categories described above, the Model Optimizer fails with an appropriate message and a link to the corresponding question in [Model Optimizer FAQ](../Model_Optimizer_FAQ.md).

The general process is as shown:

![Example custom layer network](../../img/mo_caffe_priorities.png)

1.  The example model is fed to the Model Optimizer that **loads the model** with the special parser, built on top of `caffe.proto` file. In case of failure, Model Optimizer asks you to prepare the parser that can read the model. For more information, refer to Model Optimizer, <a href="MO_FAQ.html#FAQ1">FAQ #1</a>.

2.  Model Optimizer **extracts the attributes of all layers**. In particular, it goes through the list of layers and attempts to find the appropriate extractor. In order of priority, Model Optimizer checks if the layer is:
    
    *   Registered in `CustomLayersMapping.xml`
    *   Registered as a Model Optimizer extension
    *   Registered as a standard Model Optimizer layer
    
    When the Model Optimizer finds a satisfying condition from the list above, it extracts the attributes according to the following rules:
    
    *   For bullet #1 - either takes all parameters or no parameters, according to the content of `CustomLayersMapping.xml`
    *   For bullet #2 - takes only the parameters specified in the extension
    *   For bullet #3 - takes only the parameters specified in the standard extractor
	
3.  Model Optimizer **calculates the output shape of all layers**. The logic is the same as it is for the priorities. **Important:** the Model Optimizer always takes the first available option.

4.  Model Optimizer **optimizes the original model and produces the Intermediate Representation**.

## TensorFlow\* Models with Custom Layers <a name="Tensorflow-models-with-custom-layers"></a>

You have two options for TensorFlow\* models with custom layers:

*   **Register those layers as extensions to the Model Optimizer.** In this case, the Model Optimizer generates a valid and optimized Intermediate Representation.
*   **If you have sub-graphs that should not be expressed with the analogous sub-graph in the Intermediate Representation, but another sub-graph should appear in the model, the Model Optimizer provides such an option.** This feature is helpful for many TensorFlow models. To read more, see [Sub-graph Replacement in the Model Optimizer](Subgraph_Replacement_Model_Optimizer.md).
	
## MXNet\* Models with Custom Layers <a name="mxnet-models-with-custom-layers"></a>

There are two options to convert your MXNet* model that contains custom layers:

1.  Register the custom layers as extensions to the Model Optimizer. For instructions, see [Extending MXNet Model Optimizer with New Primitives](Extending_MXNet_Model_Optimizer_with_New_Primitives.md). When your custom layers are registered as extensions, the Model Optimizer generates a valid and optimized Intermediate Representation. You can create Model Optimizer extensions for both MXNet layers with op `Custom` and layers which are not standard MXNet layers.

2.  If you have sub-graphs that should not be expressed with the analogous sub-graph in the Intermediate Representation, but another sub-graph should appear in the model, the Model Optimizer provides such an option. In MXNet the function is actively used for ssd models provides an opportunity to  for the necessary subgraph sequences and replace them. To read more, see [Sub-graph Replacement in the Model Optimizer](Subgraph_Replacement_Model_Optimizer.md).

