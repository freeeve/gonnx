package gather

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var gatherTypeConstraints = [][]tensor.Dtype{
	ops.AllTypes,
	{tensor.Int32, tensor.Int64},
}

// Gather represents the ONNX gather operator.
type Gather struct {
	ops.BaseOperator

	axis int // axis to gather on, default is 0
}

// newGather creates a new gather operator.
func newGather(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Gather{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"gather",
		),
		axis: 0,
	}
}

// Init initializes the gather operator.
func (g *Gather) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()

	if len(attributes) == 1 {
		attr := attributes[0]

		if attr.GetName() == axis {
			g.axis = int(attr.GetI())
		} else {
			return ops.ErrInvalidAttribute(attr.GetName(), g)
		}
	} else if len(attributes) > 1 {
		return ops.ErrInvalidAttributeCount(1, len(attributes), g)
	}

	return nil
}

// Apply applies the gather operator.
func (g *Gather) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	indicesData, err := ops.AnyToIntSlice(ops.IfScalarToSlice(inputs[1].Data()))
	if err != nil {
		return nil, err
	}

	data := inputs[0]

	rank := len(data.Shape())
	dataAxis := g.axis

	if dataAxis < -rank || dataAxis > rank-1 {
		return nil, ops.ErrAxisOutOfRange(rank, rank, dataAxis)
	}

	if dataAxis < 0 {
		dataAxis += rank
	}

	axisDimSize := data.Shape()[dataAxis]
	if !ops.AllInRange(indicesData, -axisDimSize, axisDimSize-1) {
		return nil, ops.ErrNotAllAxesInRange(axisDimSize, axisDimSize)
	}

	// Offset negative indices in place.
	ops.OffsetArrayIfNegative(indicesData, axisDimSize)

	indicesShape := inputs[1].Shape()
	os := insertWithReplace(indicesShape, data.Shape(), dataAxis)
	output := tensor.New(tensor.WithShape(os...), tensor.Of(data.Dtype()))

	// Fast path for axis=0: direct backing array copy.
	// Skip for scalar outputs (empty shape) since Data() returns a scalar, not a slice.
	if dataAxis == 0 && rank >= 1 && len(os) > 0 {
		if err := gatherAxis0(output, data, indicesData); err != nil {
			return nil, err
		}

		return []tensor.Tensor{output}, nil
	}

	// General path for arbitrary axis.
	indices := tensor.New(tensor.WithBacking(indicesData), tensor.WithShape(indicesShape...))

	err = gather(output, data, indices, dataAxis)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{output}, nil
}

// gatherAxis0 performs gather on axis=0 using direct backing array copy.
// Each index k selects a contiguous block of innerSize elements from the data backing array.
func gatherAxis0(out, data tensor.Tensor, indices []int) error {
	dataShape := data.Shape()

	innerSize := 1
	for i := 1; i < len(dataShape); i++ {
		innerSize *= dataShape[i]
	}

	switch src := data.Data().(type) {
	case []float32:
		dst := out.Data().([]float32)
		gatherAxis0Typed(dst, src, indices, innerSize)
	case []float64:
		dst := out.Data().([]float64)
		gatherAxis0Typed(dst, src, indices, innerSize)
	case []int:
		dst := out.Data().([]int)
		gatherAxis0Typed(dst, src, indices, innerSize)
	case []int8:
		dst := out.Data().([]int8)
		gatherAxis0Typed(dst, src, indices, innerSize)
	case []int16:
		dst := out.Data().([]int16)
		gatherAxis0Typed(dst, src, indices, innerSize)
	case []int32:
		dst := out.Data().([]int32)
		gatherAxis0Typed(dst, src, indices, innerSize)
	case []int64:
		dst := out.Data().([]int64)
		gatherAxis0Typed(dst, src, indices, innerSize)
	case []uint8:
		dst := out.Data().([]uint8)
		gatherAxis0Typed(dst, src, indices, innerSize)
	case []uint16:
		dst := out.Data().([]uint16)
		gatherAxis0Typed(dst, src, indices, innerSize)
	case []uint32:
		dst := out.Data().([]uint32)
		gatherAxis0Typed(dst, src, indices, innerSize)
	case []uint64:
		dst := out.Data().([]uint64)
		gatherAxis0Typed(dst, src, indices, innerSize)
	case []bool:
		dst := out.Data().([]bool)
		gatherAxis0Typed(dst, src, indices, innerSize)
	default:
		// Fallback to the general slice-based path.
		idxTensor := tensor.New(tensor.WithBacking(indices), tensor.WithShape(len(indices)))
		return gather(out, data, idxTensor, 0)
	}

	return nil
}

func gatherAxis0Typed[T any](dst, src []T, indices []int, innerSize int) {
	for i, k := range indices {
		srcOff := k * innerSize
		dstOff := i * innerSize
		copy(dst[dstOff:dstOff+innerSize], src[srcOff:srcOff+innerSize])
	}
}

// gather implements the general gather for any axis using slice-based iteration.
// Slice allocations are hoisted outside the loop and reused across iterations.
func gather(out, data, indices tensor.Tensor, axis int) error {
	dataRank := len(data.Shape())
	indicesRank := len(indices.Shape())
	osliceLen := indicesRank + dataRank - 1

	// Pre-allocate slice arrays and reusable slicers.
	dslices := make([]tensor.Slice, dataRank)
	oslices := make([]tensor.Slice, osliceLen)
	dSlicer := &ops.Slicer{}
	oSlicers := make([]ops.Slicer, indicesRank)

	it := indices.Iterator()
	it.Reset()

	for !it.Done() {
		coords := it.Coord()

		at, err := indices.At(coords...)
		if err != nil {
			return err
		}

		k, ok := at.(int)
		if !ok {
			return ops.ErrTypeAssert("int", at)
		}

		// Reset dslices and set the axis slicer.
		for i := range dslices {
			dslices[i] = nil
		}

		dSlicer.SetStartEnd(k, k+1)
		dslices[axis] = dSlicer

		dataSlice, _ := data.Slice(dslices...)

		// Reset oslices and set the coordinate slicers.
		for i := range oslices {
			oslices[i] = nil
		}

		for i, s := range coords {
			oSlicers[i].SetStartEnd(s, s+1)
			oslices[i+axis] = &oSlicers[i]
		}

		outputSlice, _ := out.Slice(oslices...)

		err = ops.PairwiseAssign(outputSlice, dataSlice)
		if err != nil {
			return err
		}

		_, err = it.Next()
		if err != nil {
			return err
		}
	}

	return nil
}

// insertWithReplace makes a new array, which is equal to an insertion of all elements of `a`
// into `x` at index `axis`. The element at x[axis] is removed (i.e. it is replaced with `a`).
// Output array always has length: len(a) + len(x) - 1
// Example:
// > a = [-1, -2, -3]
// > x = [1, 2, 3, 4, 5, 6, 7]
// insertWithReplace(a, x, 3) -> [1, 2, 3, -1, -2, -3, 5, 6, 7].
func insertWithReplace(a, x []int, axis int) []int {
	y := append([]int{}, x[:axis]...)
	y = append(y, a...)

	if axis+1 < len(x) {
		y = append(y, x[axis+1:]...)
	}

	return y
}
