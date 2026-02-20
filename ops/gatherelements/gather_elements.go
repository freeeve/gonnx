package gatherelements

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var gatherElementsTypeConstraints = [][]tensor.Dtype{
	ops.AllTypes,
	{tensor.Int32, tensor.Int64},
}

// GatherElements represents the ONNX GatherElements operator.
type GatherElements struct {
	ops.BaseOperator

	axis int
}

// newGatherElements creates a new GatherElements operator.
func newGatherElements(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &GatherElements{
		BaseOperator: ops.NewBaseOperator(
			version,
			2,
			2,
			typeConstraints,
			"gatherelements",
		),
		axis: 0,
	}
}

// Init initializes the GatherElements operator.
func (g *GatherElements) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "axis":
			g.axis = int(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), g)
		}
	}

	return nil
}

// Apply applies the GatherElements operator.
// output[i][j][k] = input[index[i][j][k]][j][k]  // if axis == 0
// output[i][j][k] = input[i][index[i][j][k]][k]  // if axis == 1
// output[i][j][k] = input[i][j][index[i][j][k]]  // if axis == 2
func (g *GatherElements) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	data := inputs[0]
	indices := inputs[1]

	rank := len(data.Shape())
	axis := g.axis
	if axis < 0 {
		axis += rank
	}

	axisDimSize := data.Shape()[axis]

	// Output has same shape as indices.
	output := tensor.New(tensor.WithShape(indices.Shape()...), tensor.Of(data.Dtype()))

	// Convert indices to int slice.
	indicesData, err := ops.AnyToIntSlice(ops.IfScalarToSlice(indices.Data()))
	if err != nil {
		return nil, err
	}

	indicesTensor := tensor.New(tensor.WithBacking(indicesData), tensor.WithShape(indices.Shape()...))

	it := indicesTensor.Iterator()
	it.Reset()

	for !it.Done() {
		coords := it.Coord()

		// Get the index value at these coordinates.
		idxVal, err := indicesTensor.At(coords...)
		if err != nil {
			return nil, err
		}

		idx, ok := idxVal.(int)
		if !ok {
			return nil, ops.ErrTypeAssert("int", idxVal)
		}

		// Handle negative indices.
		if idx < 0 {
			idx += axisDimSize
		}

		// Build the source coordinates: same as output coords but replace the axis dimension.
		srcCoords := make([]int, len(coords))
		copy(srcCoords, coords)
		srcCoords[axis] = idx

		val, err := data.At(srcCoords...)
		if err != nil {
			return nil, err
		}

		err = output.SetAt(val, coords...)
		if err != nil {
			return nil, err
		}

		_, err = it.Next()
		if err != nil {
			return nil, err
		}
	}

	return []tensor.Tensor{output}, nil
}
