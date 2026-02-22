package where

import (
	"fmt"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var whereTypeConstraints = [][]tensor.Dtype{
	{tensor.Bool},
	ops.AllTypes,
	ops.AllTypes,
}

// Where represents the ONNX where operator.
type Where struct {
	ops.BaseOperator
}

// newWhere creates a new where operator.
func newWhere(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Where{
		BaseOperator: ops.NewBaseOperator(
			version,
			3,
			3,
			typeConstraints,
			"where",
		),
	}
}

// Init initializes the where operator.
func (w *Where) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the where operator.
func (w *Where) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	condition := inputs[0]

	X := inputs[1]
	Y := inputs[2]

	X, Y, err := ops.MultidirectionalBroadcast(X, Y)
	if err != nil {
		return nil, err
	}

	condition, X, err = ops.MultidirectionalBroadcast(condition, X)
	if err != nil {
		return nil, err
	}

	out, err := where(X, Y, condition)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{out}, err
}

// where selects elements from X or Y based on the condition tensor using direct backing array access.
func where(X, Y, condition tensor.Tensor) (tensor.Tensor, error) {
	out := tensor.New(tensor.Of(X.Dtype()), tensor.WithShape(X.Shape()...))
	cond := condition.Data().([]bool)

	switch X.Dtype() {
	case tensor.Float32:
		whereTyped(out.Data().([]float32), X.Data().([]float32), Y.Data().([]float32), cond)
	case tensor.Float64:
		whereTyped(out.Data().([]float64), X.Data().([]float64), Y.Data().([]float64), cond)
	case tensor.Int8:
		whereTyped(out.Data().([]int8), X.Data().([]int8), Y.Data().([]int8), cond)
	case tensor.Int16:
		whereTyped(out.Data().([]int16), X.Data().([]int16), Y.Data().([]int16), cond)
	case tensor.Int32:
		whereTyped(out.Data().([]int32), X.Data().([]int32), Y.Data().([]int32), cond)
	case tensor.Int64:
		whereTyped(out.Data().([]int64), X.Data().([]int64), Y.Data().([]int64), cond)
	case tensor.Uint8:
		whereTyped(out.Data().([]uint8), X.Data().([]uint8), Y.Data().([]uint8), cond)
	case tensor.Uint16:
		whereTyped(out.Data().([]uint16), X.Data().([]uint16), Y.Data().([]uint16), cond)
	case tensor.Uint32:
		whereTyped(out.Data().([]uint32), X.Data().([]uint32), Y.Data().([]uint32), cond)
	case tensor.Uint64:
		whereTyped(out.Data().([]uint64), X.Data().([]uint64), Y.Data().([]uint64), cond)
	case tensor.Complex64:
		whereTyped(out.Data().([]complex64), X.Data().([]complex64), Y.Data().([]complex64), cond)
	case tensor.Complex128:
		whereTyped(out.Data().([]complex128), X.Data().([]complex128), Y.Data().([]complex128), cond)
	case tensor.Bool:
		whereTyped(out.Data().([]bool), X.Data().([]bool), Y.Data().([]bool), cond)
	case tensor.String:
		whereTyped(out.Data().([]string), X.Data().([]string), Y.Data().([]string), cond)
	default:
		return nil, fmt.Errorf("unsupported dtype for where: %v", X.Dtype())
	}

	return out, nil
}

func whereTyped[T any](dst, x, y []T, cond []bool) {
	for i, c := range cond {
		if c {
			dst[i] = x[i]
		} else {
			dst[i] = y[i]
		}
	}
}
