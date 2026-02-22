package erf

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var erfTypeConstraints = [][]tensor.Dtype{ops.NumericTypes}

// Erf represents the ONNX erf operator.
type Erf struct {
	ops.BaseOperator
}

// newSin creates a new erf operator.
func newErf(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Erf{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"erf",
		),
	}
}

// Init initializes the erf operator.
func (e *Erf) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the erf operator using direct backing array manipulation.
func (e *Erf) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := inputs[0]

	switch input.Dtype() {
	case tensor.Uint8:
		return erfTyped[uint8](input)
	case tensor.Uint16:
		return erfTyped[uint16](input)
	case tensor.Uint32:
		return erfTyped[uint32](input)
	case tensor.Uint64:
		return erfTyped[uint64](input)
	case tensor.Int8:
		return erfTyped[int8](input)
	case tensor.Int16:
		return erfTyped[int16](input)
	case tensor.Int32:
		return erfTyped[int32](input)
	case tensor.Int64:
		return erfTyped[int64](input)
	case tensor.Float32:
		return erfTyped[float32](input)
	case tensor.Float64:
		return erfTyped[float64](input)
	default:
		return nil, ops.ErrInvalidInputType(0, input.Dtype().String(), e.BaseOperator)
	}
}

func erfTyped[T ops.NumericType](input tensor.Tensor) ([]tensor.Tensor, error) {
	data := input.Data().([]T)
	out := make([]T, len(data))
	for i, v := range data {
		out[i] = T(math.Erf(float64(v)))
	}
	t := tensor.New(tensor.WithBacking(out), tensor.WithShape(input.Shape()...))
	return []tensor.Tensor{t}, nil
}
