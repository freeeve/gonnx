package layernormalization

import (
	"math"
	"testing"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestLayerNormalizationInit(t *testing.T) {
	l := &LayerNormalization{}
	err := l.Init(&onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "axis", I: -1},
			{Name: "epsilon", F: 1e-5},
		},
	})

	assert.Nil(t, err)
	assert.Equal(t, -1, l.axis)
	assert.Equal(t, float32(1e-5), l.epsilon)
}

func TestLayerNormalization(t *testing.T) {
	// Simple test: 2x3 input, normalize over last axis.
	ln := layerNormVersions[17]()
	err := ln.Init(&onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "axis", I: -1},
			{Name: "epsilon", F: 1e-5},
		},
	})
	assert.Nil(t, err)

	x := ops.TensorWithBackingFixture([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	scale := ops.TensorWithBackingFixture([]float32{1, 1, 1}, 3)
	bias := ops.TensorWithBackingFixture([]float32{0, 0, 0}, 3)

	res, err := ln.Apply([]tensor.Tensor{x, scale, bias})
	assert.Nil(t, err)
	assert.Equal(t, 3, len(res))

	// Check output shape.
	assert.Equal(t, tensor.Shape{2, 3}, res[0].Shape())

	// Check that values are normalized: mean ~0, std ~1 for each row.
	output := res[0].Data().([]float32)
	for row := 0; row < 2; row++ {
		var sum float32
		for j := 0; j < 3; j++ {
			sum += output[row*3+j]
		}
		mean := sum / 3.0
		assert.InDelta(t, 0.0, mean, 1e-4)

		var varSum float32
		for j := 0; j < 3; j++ {
			diff := output[row*3+j] - mean
			varSum += diff * diff
		}
		std := float32(math.Sqrt(float64(varSum / 3.0)))
		assert.InDelta(t, 1.0, std, 1e-3)
	}
}

func TestLayerNormalizationWithScaleAndBias(t *testing.T) {
	ln := layerNormVersions[17]()
	err := ln.Init(&onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "axis", I: -1},
			{Name: "epsilon", F: 1e-5},
		},
	})
	assert.Nil(t, err)

	x := ops.TensorWithBackingFixture([]float32{1, 2, 3}, 1, 3)
	scale := ops.TensorWithBackingFixture([]float32{2, 2, 2}, 3)
	bias := ops.TensorWithBackingFixture([]float32{1, 1, 1}, 3)

	res, err := ln.Apply([]tensor.Tensor{x, scale, bias})
	assert.Nil(t, err)

	output := res[0].Data().([]float32)
	// Mean of [1,2,3] = 2, std ~ 0.8165
	// normalized: [-1.2247, 0, 1.2247]
	// scaled: [-2.4495, 0, 2.4495]
	// biased: [-1.4495, 1, 3.4495]
	assert.InDelta(t, -1.4495, output[0], 0.01)
	assert.InDelta(t, 1.0, output[1], 0.01)
	assert.InDelta(t, 3.4495, output[2], 0.01)
}

func TestLayerNormalizationNoBias(t *testing.T) {
	ln := layerNormVersions[17]()
	err := ln.Init(&onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "axis", I: -1},
			{Name: "epsilon", F: 1e-5},
		},
	})
	assert.Nil(t, err)

	x := ops.TensorWithBackingFixture([]float32{1, 2, 3}, 1, 3)
	scale := ops.TensorWithBackingFixture([]float32{1, 1, 1}, 3)

	res, err := ln.Apply([]tensor.Tensor{x, scale})
	assert.Nil(t, err)

	output := res[0].Data().([]float32)
	assert.InDelta(t, -1.2247, output[0], 0.01)
	assert.InDelta(t, 0.0, output[1], 0.01)
	assert.InDelta(t, 1.2247, output[2], 0.01)
}

func TestInputValidationLayerNormalization(t *testing.T) {
	tests := []struct {
		inputs      []tensor.Tensor
		err         error
		expectedLen int
	}{
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2, 3}, 3),
				ops.TensorWithBackingFixture([]float32{1, 1, 1}, 3),
			},
			nil,
			3,
		},
		{
			[]tensor.Tensor{
				ops.TensorWithBackingFixture([]float32{1, 2, 3}, 3),
			},
			ops.ErrInvalidOptionalInputCount(1, layerNorm17BaseOpFixture()),
			0,
		},
	}

	for _, test := range tests {
		ln := layerNormVersions[17]()
		validated, err := ln.ValidateInputs(test.inputs)

		assert.Equal(t, test.err, err)
		if test.err == nil {
			assert.Equal(t, test.expectedLen, len(validated))
		}
	}
}

func BenchmarkLayerNormalization_Apply(b *testing.B) {
	ln := layerNormVersions[17]()
	err := ln.Init(&onnx.NodeProto{
		Attribute: []*onnx.AttributeProto{
			{Name: "axis", I: -1},
			{Name: "epsilon", F: 1e-5},
		},
	})
	if err != nil {
		b.Fatal(err)
	}

	x := ops.Float32TensorFixture(4, 64, 128)
	scale := ops.Float32TensorFixture(128)
	bias := ops.Float32TensorFixture(128)

	inputs := []tensor.Tensor{x, scale, bias}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		y, err := ln.Apply(inputs)
		if err != nil {
			b.Fatal(err)
		}
		_ = y
	}
}

func layerNorm17BaseOpFixture() ops.BaseOperator {
	return ops.NewBaseOperator(17, 2, 3, layerNormTypeConstraints, "layernormalization")
}
