package gatherelements

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestGatherElementsInit(t *testing.T) {
	g := &GatherElements{}
	err := g.Init(&onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "axis", I: 1},
		},
	})

	assert.Nil(t, err)
	assert.Equal(t, 1, g.axis)
}

func TestGatherElements(t *testing.T) {
	tests := []struct {
		name            string
		axis            int64
		dataBacking     []float32
		dataShape       []int
		indicesBacking  []int32
		indicesShape    []int
		expectedBacking []float32
		expectedShape   tensor.Shape
	}{
		{
			"axis 0, 2x2",
			0,
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]int32{0, 0, 1, 0},
			[]int{2, 2},
			[]float32{1, 2, 3, 2},
			[]int{2, 2},
		},
		{
			"axis 1, 2x2",
			1,
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]int32{0, 0, 1, 0},
			[]int{2, 2},
			[]float32{1, 1, 4, 3},
			[]int{2, 2},
		},
		{
			"axis 0, 3x3 from onnx spec",
			0,
			[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			[]int{3, 3},
			[]int32{1, 2, 0, 2, 0, 0},
			[]int{2, 3},
			[]float32{4, 8, 3, 7, 2, 3},
			[]int{2, 3},
		},
		{
			"axis 1, 3x3 from onnx spec",
			1,
			[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			[]int{3, 3},
			[]int32{2, 0, 0, 2},
			[]int{2, 2},
			[]float32{3, 1, 4, 6},
			[]int{2, 2},
		},
		{
			"negative index",
			0,
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]int32{-1, -2},
			[]int{1, 2},
			[]float32{3, 2},
			[]int{1, 2},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ge := gatherElementsVersions[13]()
			err := ge.Init(&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "axis", I: test.axis},
				},
			})
			assert.Nil(t, err)

			inputs := []tensor.Tensor{
				ops.TensorWithBackingFixture(test.dataBacking, test.dataShape...),
				ops.TensorWithBackingFixture(test.indicesBacking, test.indicesShape...),
			}

			res, err := ge.Apply(inputs)
			assert.Nil(t, err)
			assert.Equal(t, test.expectedShape, res[0].Shape())
			assert.Equal(t, test.expectedBacking, res[0].Data())
		})
	}
}

func TestInputValidationGatherElements(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2, 3, 4}, 2, 2),
				ops.TensorWithBackingFixture([]int32{0, 1}, 1, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			ops.ErrInvalidInputCount(1, gatherElements13BaseOpFixture()),
		},
	}

	for _, test := range tests {
		ge := gatherElementsVersions[13]()
		validated, err := ge.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func gatherElements13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 2, 2, gatherElementsTypeConstraints, "gatherelements")
}
