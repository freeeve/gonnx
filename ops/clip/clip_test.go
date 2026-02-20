package clip

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestClipInit(t *testing.T) {
	c := &Clip{}
	err := c.Init(ops.EmptyNodeProto())
	assert.Nil(t, err)
}

func TestClip(t *testing.T) {
	tests := []struct {
		name            string
		inputs          []tensor.Tensor
		expectedBacking []float32
	}{
		{
			"clip with min and max",
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{-2, -1, 0, 1, 2, 3}, 6),
				ops.TensorWithBackingFixture([]float32{-1}, 1),
				ops.TensorWithBackingFixture([]float32{2}, 1),
			},
			[]float32{-1, -1, 0, 1, 2, 2},
		},
		{
			"clip with min only",
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{-2, -1, 0, 1, 2, 3}, 6),
				ops.TensorWithBackingFixture([]float32{0}, 1),
			},
			[]float32{0, 0, 0, 1, 2, 3},
		},
		{
			"clip with no min or max",
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{-2, -1, 0, 1, 2, 3}, 6),
			},
			[]float32{-2, -1, 0, 1, 2, 3},
		},
		{
			"clip with nil min and max",
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{-2, -1, 0, 1, 2, 3}, 6),
				nil,
				ops.TensorWithBackingFixture([]float32{1.5}, 1),
			},
			[]float32{-2, -1, 0, 1, 1.5, 1.5},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			clip := clipVersions[13]()
			err := clip.Init(ops.EmptyNodeProto())
			assert.Nil(t, err)

			res, err := clip.Apply(test.inputs)
			assert.Nil(t, err)
			assert.Equal(t, test.expectedBacking, res[0].Data())
		})
	}
}

func TestInputValidationClip(t *testing.T) {
	tests := []struct {
		inputs      []tensor.Tensor
		err         error
		expectedLen int
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
			3,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
				ops.TensorWithBackingFixture([]float32{0}, 1),
				ops.TensorWithBackingFixture([]float32{3}, 1),
			},
			nil,
			3,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", clip13BaseOpFixture()),
			0,
		},
	}

	for _, test := range tests {
		clip := clipVersions[13]()
		validated, err := clip.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.expectedLen, len(validated))
		}
	}
}

func clip13BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(13, 1, 3, clipTypeConstraints, "clip")
}
