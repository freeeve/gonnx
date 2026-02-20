package reducesum

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var reduceSumTypeConstraints = [][]tensor.Dtype{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
}

var reduceSumV13TypeConstraints = [][]tensor.Dtype{
	{tensor.Uint32, tensor.Uint64, tensor.Int32, tensor.Int64, tensor.Float32, tensor.Float64},
	{tensor.Int64},
}

// ReduceSum represents the ONNX ReduceSum operator.
type ReduceSum struct {
	ops.BaseOperator

	axes              []int
	keepDims          bool
	noopWithEmptyAxes bool
}

// newReduceSum creates a new ReduceSum operator for opset < 13 (axes as attribute).
func newReduceSum(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &ReduceSum{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			1,
			typeConstraints,
			"reducesum",
		),
		axes:     []int{},
		keepDims: true,
	}
}

// newReduceSum13 creates a new ReduceSum operator for opset >= 13 (axes as input).
func newReduceSum13(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &ReduceSum{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			2,
			typeConstraints,
			"reducesum",
		),
		axes:              []int{},
		keepDims:          true,
		noopWithEmptyAxes: false,
	}
}

// Init initializes the ReduceSum operator.
func (r *ReduceSum) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "axes":
			axes, err := ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return err
			}
			r.axes = axes
		case "keepdims":
			r.keepDims = attr.GetI() == 1
		case "noop_with_empty_axes":
			r.noopWithEmptyAxes = attr.GetI() == 1
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), r)
		}
	}

	return nil
}

// Apply applies the ReduceSum operator.
func (r *ReduceSum) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := tensor.New(tensor.WithBacking(inputs[0].Data()), tensor.WithShape(inputs[0].Shape()...))

	axes := r.axes

	// For opset >= 13, axes come from the second input tensor.
	if r.Version() >= 13 && len(inputs) > 1 && inputs[1] != nil {
		// Check if axes tensor is non-empty before reading data.
		axisSize := 1
		for _, s := range inputs[1].Shape() {
			axisSize *= s
		}
		if axisSize > 0 {
			axesData, err := ops.AnyToIntSlice(ops.IfScalarToSlice(inputs[1].Data()))
			if err != nil {
				return nil, err
			}
			axes = axesData
		}
	}

	// If axes is empty and noopWithEmptyAxes, return input as-is.
	if len(axes) == 0 && r.noopWithEmptyAxes {
		return []tensor.Tensor{input}, nil
	}

	// If axes is empty and not noopWithEmptyAxes, reduce over all axes.
	if len(axes) == 0 {
		axes = make([]int, len(input.Shape()))
		for i := range axes {
			axes[i] = i
		}
	}

	// Convert negative axes.
	resolvedAxes := make([]int, len(axes))
	for i, axis := range axes {
		resolvedAxes[i] = ops.ConvertNegativeAxis(axis, len(input.Shape()))
	}

	sum, err := input.Sum(resolvedAxes...)
	if err != nil {
		return nil, err
	}

	if r.keepDims {
		newShape := make([]int, len(input.Shape()))
		copy(newShape, input.Shape())
		for _, axis := range resolvedAxes {
			newShape[axis] = 1
		}

		err := sum.Reshape(newShape...)
		if err != nil {
			return nil, err
		}
	}

	return []tensor.Tensor{sum}, nil
}
