package ops

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestTanhActivation(t *testing.T) {
	tIn := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, 2, 3, 4}))
	tOut, err := Tanh(tIn)

	assert.Nil(t, err)
	assert.Equal(t, []float32{0.7615942, 0.9640276, 0.9950548, 0.9993293}, tOut.Data())
}

func TestReLUActivation(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		expected []float32
	}{
		{
			"all negative",
			[]float32{-4, -3, -2, -1},
			[]float32{0, 0, 0, 0},
		},
		{
			"mixed",
			[]float32{-2, -1, 0, 1, 2, 3},
			[]float32{0, 0, 0, 1, 2, 3},
		},
		{
			"all positive",
			[]float32{1, 2, 3, 4},
			[]float32{1, 2, 3, 4},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tIn := tensor.New(tensor.WithShape(len(test.input)), tensor.WithBacking(test.input))
			tOut, err := ReLU(tIn)

			assert.Nil(t, err)
			assert.Equal(t, test.expected, tOut.Data())

			// Verify input is not modified.
			assert.Equal(t, test.input, tIn.Data())
		})
	}
}

func TestReLUActivationFloat64(t *testing.T) {
	tIn := tensor.New(tensor.WithShape(4), tensor.WithBacking([]float64{-2, -1, 1, 2}))
	tOut, err := ReLU(tIn)

	assert.Nil(t, err)
	assert.Equal(t, []float64{0, 0, 1, 2}, tOut.Data())
}

func TestSigmoidActivation(t *testing.T) {
	tIn := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, 2, 3, 4}))
	tOut, err := Sigmoid(tIn)

	assert.Nil(t, err)
	assert.Equal(t, []float32{0.7310586, 0.880797, 0.95257413, 0.98201376}, tOut.Data())
}
