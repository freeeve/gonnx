package leakyrelu

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var leakyReluTypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64},
}

// LeakyRelu represents the ONNX LeakyRelu operator.
type LeakyRelu struct {
	ops.BaseOperator

	alpha float32
}

// newLeakyRelu creates a new LeakyRelu operator.
func newLeakyRelu(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &LeakyRelu{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"leakyrelu",
		),
		alpha: 0.01,
	}
}

// Init initializes the LeakyRelu operator.
func (l *LeakyRelu) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "alpha":
			l.alpha = attr.GetF()
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), l)
		}
	}

	return nil
}

// Apply applies the LeakyRelu operator.
func (l *LeakyRelu) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := inputs[0]
	out := tensor.New(tensor.WithShape(input.Shape()...), tensor.Of(input.Dtype()))

	switch input.Dtype() {
	case tensor.Float32:
		err := leakyReluTyped[float32](out.Data(), input.Data(), float32(l.alpha))
		if err != nil {
			return nil, err
		}
	case tensor.Float64:
		err := leakyReluTyped[float64](out.Data(), input.Data(), float64(l.alpha))
		if err != nil {
			return nil, err
		}
	default:
		return nil, ops.ErrInvalidInputType(0, input.Dtype().String(), l.BaseOperator)
	}

	return []tensor.Tensor{out}, nil
}

func leakyReluTyped[T float32 | float64](result, input any, alpha T) error {
	res, ok := result.([]T)
	if !ok {
		return ops.ErrTypeAssert("numeric list", result)
	}

	in, ok := input.([]T)
	if !ok {
		return ops.ErrTypeAssert("numeric list", input)
	}

	for i, v := range in {
		if v < 0 {
			res[i] = alpha * v
		} else {
			res[i] = v
		}
	}

	return nil
}
