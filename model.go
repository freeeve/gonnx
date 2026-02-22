package gonnx

import (
	"archive/zip"
	"io"
	"os"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"google.golang.org/protobuf/proto"
	"gorgonia.org/tensor"
)

// Tensors is a map with tensors.
type Tensors map[string]tensor.Tensor

// cachedNode holds a pre-initialized operator and the expected input count
// for a single graph node, avoiding per-Run allocation of operators and input slices.
type cachedNode struct {
	op      ops.Operator
	nInputs int
}

// Model defines a model that can be used for inference.
type Model struct {
	mp         *onnx.ModelProto
	parameters Tensors
	Opset      Opset
	nodes      []*onnx.NodeProto
	cache      []cachedNode
}

// NewModelFromFile creates a new model from a path to a file.
func NewModelFromFile(path string) (*Model, error) {
	bytesModel, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	return NewModelFromBytes(bytesModel)
}

// NewModelFromZipFile creates a new model from a file in a zip archive.
func NewModelFromZipFile(file *zip.File) (*Model, error) {
	fc, err := file.Open()
	if err != nil {
		return nil, err
	}

	bytesModel, err := io.ReadAll(fc)
	if err != nil {
		return nil, err
	}

	return NewModelFromBytes(bytesModel)
}

// NewModelFromBytes creates a new model from a list of bytes.
func NewModelFromBytes(bytesModel []byte) (*Model, error) {
	mp, err := ModelProtoFromBytes(bytesModel)
	if err != nil {
		return nil, err
	}

	return NewModel(mp)
}

// NewModel creates a new model ready for inference given a path to an onnx file.
func NewModel(mp *onnx.ModelProto) (*Model, error) {
	params, err := mp.Graph.Params()
	if err != nil {
		return nil, err
	}

	opsetImports := mp.GetOpsetImport()

	var opsetID int64

	for i := 0; i < len(opsetImports); i++ {
		version := opsetImports[i].GetVersion()
		if version > opsetID {
			opsetID = version
		}
	}

	opset, err := ResolveOpset(opsetID)
	if err != nil {
		return nil, err
	}

	nodes := mp.Graph.GetNode()
	cache := make([]cachedNode, len(nodes))

	for i, n := range nodes {
		factory, ok := opset[n.GetOpType()]
		if !ok {
			return nil, ops.ErrUnknownOperatorType(n.GetOpType())
		}

		operator := factory()
		if err := operator.Init(n); err != nil {
			return nil, err
		}

		cache[i] = cachedNode{
			op:      operator,
			nInputs: len(n.GetInput()),
		}
	}

	return &Model{
		mp:         mp,
		parameters: params,
		Opset:      opset,
		nodes:      nodes,
		cache:      cache,
	}, nil
}

// ModelProtoFromBytes creates an onnx.ModelProto based on a list of bytes.
func ModelProtoFromBytes(bytesModel []byte) (*onnx.ModelProto, error) {
	mp := &onnx.ModelProto{}
	if err := proto.Unmarshal(bytesModel, mp); err != nil {
		return nil, err
	}

	return mp, nil
}

// InputNames returns this models input names as defined by the model proto.
func (m *Model) InputNames() []string {
	return m.mp.Graph.InputNames()
}

// InputShapes returns the shapes for all input tensors.
func (m *Model) InputShapes() onnx.Shapes {
	return m.mp.Graph.InputShapes()
}

// InputDimSize returns the size of the input dimension given an input tensor.
func (m *Model) InputDimSize(input string, i int) (int, error) {
	if !m.hasInput(input) {
		return 0, ErrModel("input %v does not exist", input)
	}

	inputShape := m.mp.Graph.InputShapes()[input]

	if i >= len(inputShape) {
		return 0, ErrModel("input %v only has %d dimensions, but index %d was required", input, len(inputShape), i)
	}

	return int(inputShape[i].Size), nil
}

// OutputNames returns this models output names as defined by the model proto.
func (m *Model) OutputNames() []string {
	return m.mp.Graph.OutputNames()
}

// OutputShapes returns the shapes for all output tensors.
func (m *Model) OutputShapes() onnx.Shapes {
	return m.mp.Graph.OutputShapes()
}

// OutputShape returns the shape of a specific output tensors.
func (m *Model) OutputShape(output string) onnx.Shape {
	return m.mp.Graph.OutputShapes()[output]
}

// ParamNames returns this models parameter names as defined by the model proto.
func (m *Model) ParamNames() []string {
	return m.mp.Graph.ParamNames()
}

func (m *Model) hasInput(input string) bool {
	for _, inputName := range m.InputNames() {
		if inputName == input {
			return true
		}
	}

	return false
}

// Run builds and executes the computional graph of the network given the inputs.
func (m *Model) Run(inputs Tensors) (Tensors, error) {
	if err := m.validateShapes(inputs); err != nil {
		return nil, err
	}

	tensors := make(Tensors, len(inputs)+len(m.parameters)+len(m.nodes))

	for inputName, inputTensor := range inputs {
		tensors[inputName] = inputTensor
	}

	for parameterName, parameterTensor := range m.parameters {
		tensors[parameterName] = parameterTensor
	}

	for i, n := range m.nodes {
		cn := &m.cache[i]
		if err := m.applyCachedOp(cn, n, tensors); err != nil {
			return nil, err
		}
	}

	outputNames := m.OutputNames()
	outputTensors := make(Tensors, len(outputNames))

	for _, outputName := range outputNames {
		outputTensors[outputName] = tensors[outputName]
	}

	return outputTensors, nil
}

// applyCachedOp applies a pre-initialized operator to the graph.
func (m *Model) applyCachedOp(cn *cachedNode, n *onnx.NodeProto, tensors Tensors) error {
	inputTensors := make([]tensor.Tensor, 0, cn.nInputs)

	for _, tensorName := range n.GetInput() {
		if tensorName == "" {
			inputTensors = append(inputTensors, nil)
		} else if t, ok := tensors[tensorName]; ok {
			inputTensors = append(inputTensors, t)
		} else {
			return ErrModel("no tensor yet for name %v", tensorName)
		}
	}

	inputTensors, err := cn.op.ValidateInputs(inputTensors)
	if err != nil {
		return err
	}

	outputTensors, err := cn.op.Apply(inputTensors)
	if err != nil {
		return err
	}

	return setOutputTensorsOfNode(n.GetOutput(), outputTensors, tensors)
}

// validateShapes validates if the tensors passed in have the same shape as the shapes defined
// by the onnx.Shapes.
func (m *Model) validateShapes(inputTensors Tensors) error {
	for name, shapeExpected := range m.InputShapes() {
		// If the input is a parameter, the user does not have to provide a tensor for it.
		if _, ok := m.parameters[name]; ok {
			continue
		}

		tensor, ok := inputTensors[name]
		if !ok {
			return ErrModel("tensor: %v not found", name)
		}

		shapeReceived := tensor.Shape()

		if len(shapeReceived) != len(shapeExpected) {
			return ErrInvalidShape(shapeExpected, shapeReceived)
		}

		for i, dim := range shapeExpected {
			// because the dimension is dynamic, it can have any size
			// and we do not have to check for it
			if dim.IsDynamic {
				continue
			}

			if dim.Size != int64(shapeReceived[i]) {
				return ErrInvalidShape(shapeExpected, shapeReceived)
			}
		}
	}

	return nil
}

func setOutputTensorsOfNode(
	names []string, outputTensors []tensor.Tensor, tensors Tensors,
) error {
	// Some operators (e.g. LayerNormalization) produce optional outputs.
	// When the model graph declares fewer output names than the operator
	// returns, we truncate the extra outputs rather than failing.
	if len(outputTensors) > len(names) {
		outputTensors = outputTensors[:len(names)]
	}
	if len(names) != len(outputTensors) {
		return ErrModel("could not set output tensor")
	}

	for i, tensor := range outputTensors {
		tensors[names[i]] = tensor
	}

	return nil
}
