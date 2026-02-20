package leakyrelu

import (
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestLeakyReluInit(t *testing.T) {
	l := &LeakyRelu{}
	err := l.Init(&onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "alpha", F: 0.2},
		},
	})

	assert.Nil(t, err)
	assert.Equal(t, float32(0.2), l.alpha)
}

func TestLeakyReluDefaultAlpha(t *testing.T) {
	l := newLeakyRelu(16, leakyReluTypeConstraints)
	err := l.Init(ops.EmptyNodeProto())
	assert.Nil(t, err)

	inputs := []tensor.Tensor{
		ops.TensorWithBackingFixture([]float32{-2, -1, 0, 1, 2}, 5),
	}

	res, err := l.Apply(inputs)
	assert.Nil(t, err)
	assert.Equal(t, []float32{-0.02, -0.01, 0, 1, 2}, res[0].Data())
}

func TestLeakyRelu(t *testing.T) {
	tests := []struct {
		name            string
		alpha           float32
		backing         []float32
		shape           []int
		expectedBacking []float32
	}{
		{
			"alpha 0.1",
			0.1,
			[]float32{-3, -2, -1, 0, 1, 2},
			[]int{6},
			[]float32{-0.3, -0.2, -0.1, 0, 1, 2},
		},
		{
			"alpha 0.2",
			0.2,
			[]float32{-1, 0, 1},
			[]int{3},
			[]float32{-0.2, 0, 1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			leaky := leakyReluVersions[16]()
			err := leaky.Init(&onnx.NodeProto{
				Attribute: []*onnx.AttributeProto{
					{Name: "alpha", F: test.alpha},
				},
			})
			assert.Nil(t, err)

			inputs := []tensor.Tensor{
				ops.TensorWithBackingFixture(test.backing, test.shape...),
			}

			res, err := leaky.Apply(inputs)
			assert.Nil(t, err)
			assert.Equal(t, test.expectedBacking, res[0].Data())
		})
	}
}

func TestInputValidationLeakyRelu(t *testing.T) {
	tests := []struct {
		inputs []tensor.Tensor
		err    error
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2}, 2),
			},
			nil,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]int{1, 2}, 2),
			},
			ops.ErrInvalidInputType(0, "int", leakyRelu16BaseOpFixture()),
		},
	}

	for _, test := range tests {
		leaky := leakyReluVersions[16]()
		validated, err := leaky.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.inputs, validated)
		}
	}
}

func leakyRelu16BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(16, 1, 1, leakyReluTypeConstraints, "leakyrelu")
}
