package layernormalization

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var layerNormTypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
}

// LayerNormalization represents the ONNX LayerNormalization operator.
type LayerNormalization struct {
	ops.BaseOperator

	axis       int
	epsilon    float32
	stashType  int
	numOutputs int
}

// newLayerNormalization creates a new LayerNormalization operator.
func newLayerNormalization(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &LayerNormalization{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			3,
			typeConstraints,
			"layernormalization",
		),
		axis:      -1,
		epsilon:   1e-5,
		stashType: 1,
	}
}

// Init initializes the LayerNormalization operator.
func (l *LayerNormalization) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "axis":
			l.axis = int(attr.GetI())
		case "epsilon":
			l.epsilon = attr.GetF()
		case "stash_type":
			l.stashType = int(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), l)
		}
	}

	l.numOutputs = len(n.GetOutput())
	if l.numOutputs == 0 {
		l.numOutputs = 3
	}

	return nil
}

// Apply applies the LayerNormalization operator.
// Computes: Y = (X - Mean) / Sqrt(Var + epsilon) * Scale + Bias
// Returns up to 3 outputs: Y, Mean, InvStdDev (only Y is required).
func (l *LayerNormalization) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	x := inputs[0]
	scale := inputs[1]

	var bias tensor.Tensor
	if len(inputs) > 2 && inputs[2] != nil {
		bias = inputs[2]
	}

	rank := len(x.Shape())
	axis := l.axis
	if axis < 0 {
		axis += rank
	}

	switch x.Dtype() {
	case tensor.Float32:
		return l.applyFloat32(x, scale, bias, axis)
	case tensor.Float64:
		return l.applyFloat64(x, scale, bias, axis)
	default:
		return nil, ops.ErrInvalidInputType(0, x.Dtype().String(), l.BaseOperator)
	}
}

func (l *LayerNormalization) applyFloat32(x, scale, bias tensor.Tensor, axis int) ([]tensor.Tensor, error) {
	xData := x.Data().([]float32)
	scaleData := scale.Data().([]float32)

	var biasData []float32
	if bias != nil {
		biasData = bias.Data().([]float32)
	}

	shape := x.Shape()

	// Calculate the number of elements in the outer and inner dimensions.
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= shape[i]
	}

	innerSize := 1
	for i := axis; i < len(shape); i++ {
		innerSize *= shape[i]
	}

	output := make([]float32, len(xData))
	meanData := make([]float32, outerSize)
	invStdData := make([]float32, outerSize)
	epsilon := l.epsilon

	for i := 0; i < outerSize; i++ {
		offset := i * innerSize

		// Compute mean.
		var sum float32
		for j := 0; j < innerSize; j++ {
			sum += xData[offset+j]
		}
		mean := sum / float32(innerSize)
		meanData[i] = mean

		// Compute variance.
		var varSum float32
		for j := 0; j < innerSize; j++ {
			diff := xData[offset+j] - mean
			varSum += diff * diff
		}
		variance := varSum / float32(innerSize)
		invStd := float32(1.0 / math.Sqrt(float64(variance+epsilon)))
		invStdData[i] = invStd

		// Normalize, scale, and bias.
		for j := 0; j < innerSize; j++ {
			normalized := (xData[offset+j] - mean) * invStd
			result := normalized * scaleData[j]
			if biasData != nil {
				result += biasData[j]
			}
			output[offset+j] = result
		}
	}

	yTensor := tensor.New(tensor.WithBacking(output), tensor.WithShape(shape...))

	if l.numOutputs <= 1 {
		return []tensor.Tensor{yTensor}, nil
	}

	meanShape := make([]int, axis)
	copy(meanShape, shape[:axis])
	if len(meanShape) == 0 {
		meanShape = []int{1}
	}
	meanTensor := tensor.New(tensor.WithBacking(meanData), tensor.WithShape(meanShape...))
	invStdTensor := tensor.New(tensor.WithBacking(invStdData), tensor.WithShape(meanShape...))

	return []tensor.Tensor{yTensor, meanTensor, invStdTensor}, nil
}

func (l *LayerNormalization) applyFloat64(x, scale, bias tensor.Tensor, axis int) ([]tensor.Tensor, error) {
	xData := x.Data().([]float64)
	scaleData := scale.Data().([]float64)

	var biasData []float64
	if bias != nil {
		biasData = bias.Data().([]float64)
	}

	shape := x.Shape()

	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= shape[i]
	}

	innerSize := 1
	for i := axis; i < len(shape); i++ {
		innerSize *= shape[i]
	}

	output := make([]float64, len(xData))
	meanData := make([]float64, outerSize)
	invStdData := make([]float64, outerSize)
	epsilon := float64(l.epsilon)

	for i := 0; i < outerSize; i++ {
		offset := i * innerSize

		var sum float64
		for j := 0; j < innerSize; j++ {
			sum += xData[offset+j]
		}
		mean := sum / float64(innerSize)
		meanData[i] = mean

		var varSum float64
		for j := 0; j < innerSize; j++ {
			diff := xData[offset+j] - mean
			varSum += diff * diff
		}
		variance := varSum / float64(innerSize)
		invStd := 1.0 / math.Sqrt(variance+epsilon)
		invStdData[i] = invStd

		for j := 0; j < innerSize; j++ {
			normalized := (xData[offset+j] - mean) * invStd
			result := normalized * scaleData[j]
			if biasData != nil {
				result += biasData[j]
			}
			output[offset+j] = result
		}
	}

	yTensor := tensor.New(tensor.WithBacking(output), tensor.WithShape(shape...))

	if l.numOutputs <= 1 {
		return []tensor.Tensor{yTensor}, nil
	}

	meanShape := make([]int, axis)
	copy(meanShape, shape[:axis])
	if len(meanShape) == 0 {
		meanShape = []int{1}
	}
	meanTensor := tensor.New(tensor.WithBacking(meanData), tensor.WithShape(meanShape...))
	invStdTensor := tensor.New(tensor.WithBacking(invStdData), tensor.WithShape(meanShape...))

	return []tensor.Tensor{yTensor, meanTensor, invStdTensor}, nil
}
