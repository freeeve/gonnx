package reducesum

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestReduceSumInit(t *testing.T) {
	r := &ReduceSum{}
	err := r.Init(&onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "axes", Ints: []int64{1, 3}},
			{Name: "keepdims", I: 0},
		},
	})

	assert.Nil(t, err)
	assert.Equal(t, []int{1, 3}, r.axes)
	assert.Equal(t, false, r.keepDims)
}

func TestReduceSumV11(t *testing.T) {
	tests := []struct {
		name            string
		node            *onnx.NodeProto
		backing         []float32
		shape           []int
		expectedBacking []float32
		expectedShape   tensor.Shape
	}{
		{
			"axis 0 no keepdims",
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{0}},
					{Name: "keepdims", I: 0},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{2, 4},
			[]int{2},
		},
		{
			"axis 0 keepdims",
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{0}},
					{Name: "keepdims", I: 1},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{2, 4},
			[]int{1, 2},
		},
		{
			"axis 1 no keepdims",
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{1}},
					{Name: "keepdims", I: 0},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{1, 5},
			[]int{2},
		},
		{
			"negative axis keepdims",
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axes", Ints: []int64{-1}},
					{Name: "keepdims", I: 1},
				},
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
			[]float32{1, 5},
			[]int{2, 1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			inputs := []tensor.Tensor{
				ops.TensorWithBackingFixture(test.backing, test.shape...),
			}

			rs := reduceSumVersions[11]()
			err := rs.Init(test.node)
			assert.Nil(t, err)

			res, err := rs.Apply(inputs)
			assert.Nil(t, err)
			assert.Equal(t, test.expectedShape, res[0].Shape())
			assert.Equal(t, test.expectedBacking, res[0].Data())
		})
	}
}

func TestReduceSumV13(t *testing.T) {
	tests := []struct {
		name            string
		node            *onnx.NodeProto
		inputs          []tensor.Tensor
		expectedBacking []float32
		expectedShape   tensor.Shape
	}{
		{
			"axes from input, axis 0 no keepdims",
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "keepdims", I: 0},
				},
			},
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{0, 1, 2, 3}, 2, 2),
				ops.TensorWithBackingFixture([]int64{0}, 1),
			},
			[]float32{2, 4},
			[]int{2},
		},
		{
			"axes from input, axis 1 keepdims",
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "keepdims", I: 1},
				},
			},
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{0, 1, 2, 3}, 2, 2),
				ops.TensorWithBackingFixture([]int64{1}, 1),
			},
			[]float32{1, 5},
			[]int{2, 1},
		},
		{
			"noop with empty axes",
			&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "keepdims", I: 1},
					{Name: "noop_with_empty_axes", I: 1},
				},
			},
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{0, 1, 2, 3}, 2, 2),
			},
			[]float32{0, 1, 2, 3},
			[]int{2, 2},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			rs := reduceSumVersions[13]()
			err := rs.Init(test.node)
			assert.Nil(t, err)

			res, err := rs.Apply(test.inputs)
			assert.Nil(t, err)
			assert.Equal(t, test.expectedShape, res[0].Shape())
			assert.Equal(t, test.expectedBacking, res[0].Data())
		})
	}
}

func TestInputValidationReduceSum(t *testing.T) {
	tests := []struct {
		version     int64
		inputs      []tensor.Tensor
		err         error
		expectedLen int
	}{
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			2,
		},
		{
			13,
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", reduceSum13BaseOpFixture()),
			0,
		},
	}

	for _, test := range tests {
		rs := reduceSumVersions[test.version]()
		validated, err := rs.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.expectedLen, len(validated))
		}
	}
}

func reduceSum13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 2, reduceSumV13TypeConstraints, "reducesum")
}
