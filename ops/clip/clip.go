package clip

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var clipTypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
}

// Clip represents the ONNX clip operator.
type Clip struct {
	ops.BaseOperator
}

// newClip creates a new clip operator.
func newClip(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Clip{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			3,
			typeConstraints,
			"clip",
		),
	}
}

// Init initializes the clip operator.
func (c *Clip) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the clip operator.
func (c *Clip) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := inputs[0]

	minVal := -math.MaxFloat64
	maxVal := math.MaxFloat64

	if len(inputs) > 1 && inputs[1] != nil {
		v, err := getScalarFloat64(inputs[1])
		if err != nil {
			return nil, err
		}
		minVal = v
	}

	if len(inputs) > 2 && inputs[2] != nil {
		v, err := getScalarFloat64(inputs[2])
		if err != nil {
			return nil, err
		}
		maxVal = v
	}

	out := tensor.New(tensor.WithShape(input.Shape()...), tensor.Of(input.Dtype()))

	switch input.Dtype() {
	case tensor.Float32:
		err := clipTyped[float32](out.Data(), input.Data(), float32(minVal), float32(maxVal))
		if err != nil {
			return nil, err
		}
	case tensor.Float64:
		err := clipTyped[float64](out.Data(), input.Data(), minVal, maxVal)
		if err != nil {
			return nil, err
		}
	default:
		return nil, ops.ErrInvalidInputType(0, input.Dtype().String(), c.BaseOperator)
	}

	return []tensor.Tensor{out}, nil
}

func clipTyped[T float32 | float64](result, input any, minVal, maxVal T) error {
	res, ok := result.([]T)
	if !ok {
		return ops.ErrTypeAssert("numeric list", result)
	}

	in, ok := input.([]T)
	if !ok {
		return ops.ErrTypeAssert("numeric list", input)
	}

	for i, v := range in {
		if v < minVal {
			v = minVal
		}
		if v > maxVal {
			v = maxVal
		}
		res[i] = v
	}

	return nil
}

func getScalarFloat64(t tensor.Tensor) (float64, error) {
	switch t.Dtype() {
	case tensor.Float32:
		data := t.Data()
		switch d := data.(type) {
		case []float32:
			if len(d) > 0 {
				return float64(d[0]), nil
			}
		case float32:
			return float64(d), nil
		}
	case tensor.Float64:
		data := t.Data()
		switch d := data.(type) {
		case []float64:
			if len(d) > 0 {
				return d[0], nil
			}
		case float64:
			return d, nil
		}
	}

	return 0, ops.ErrCast
}
