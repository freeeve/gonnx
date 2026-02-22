package ops

import (
	"math"

	"github.com/chewxy/math32"
	"gorgonia.org/tensor"
)

// Activation is an activation function.
type Activation func(n tensor.Tensor) (tensor.Tensor, error)

// activations maps strings to the activation function. This is
// used by operators like LSTM, GRU and RNN.
var activations = map[string]Activation{
	"tanh":    Tanh,
	"sigmoid": Sigmoid,
	"relu":    ReLU,
}

func GetActivation(activation string) (Activation, error) {
	if a, ok := activations[activation]; ok {
		return a, nil
	}

	return nil, ErrActivationNotImplemented(activation)
}

// Tanh performs the tanh operation on a tensor using direct backing array manipulation.
func Tanh(X tensor.Tensor) (tensor.Tensor, error) {
	switch X.Dtype() {
	case tensor.Float32:
		data := X.Data().([]float32)
		out := make([]float32, len(data))
		for i, v := range data {
			out[i] = math32.Tanh(v)
		}
		return tensor.New(tensor.WithBacking(out), tensor.WithShape(X.Shape()...)), nil
	case tensor.Float64:
		data := X.Data().([]float64)
		out := make([]float64, len(data))
		for i, v := range data {
			out[i] = math.Tanh(v)
		}
		return tensor.New(tensor.WithBacking(out), tensor.WithShape(X.Shape()...)), nil
	default:
		return nil, ErrCast
	}
}

// Sigmoid performs the sigmoid operation on a tensor: 1 / (1 + exp(-x)).
// Uses direct backing array manipulation to minimize allocations.
func Sigmoid(X tensor.Tensor) (tensor.Tensor, error) {
	switch X.Dtype() {
	case tensor.Float32:
		return sigmoidFloat32(X)
	case tensor.Float64:
		return sigmoidFloat64(X)
	default:
		return nil, ErrCast
	}
}

func sigmoidFloat32(X tensor.Tensor) (tensor.Tensor, error) {
	data := X.Data().([]float32)
	out := make([]float32, len(data))
	for i, v := range data {
		e := math32.Exp(-v)
		out[i] = 1.0 / (1.0 + e)
	}
	return tensor.New(tensor.WithBacking(out), tensor.WithShape(X.Shape()...)), nil
}

func sigmoidFloat64(X tensor.Tensor) (tensor.Tensor, error) {
	data := X.Data().([]float64)
	out := make([]float64, len(data))
	for i, v := range data {
		out[i] = 1.0 / (1.0 + math.Exp(-v))
	}
	return tensor.New(tensor.WithBacking(out), tensor.WithShape(X.Shape()...)), nil
}

// ReLU performs the ReLU operation on a tensor using direct backing array manipulation.
func ReLU(X tensor.Tensor) (tensor.Tensor, error) {
	switch X.Dtype() {
	case tensor.Float32:
		data := X.Data().([]float32)
		out := make([]float32, len(data))
		for i, v := range data {
			if v > 0 {
				out[i] = v
			}
		}
		return tensor.New(tensor.WithBacking(out), tensor.WithShape(X.Shape()...)), nil
	case tensor.Float64:
		data := X.Data().([]float64)
		out := make([]float64, len(data))
		for i, v := range data {
			if v > 0 {
				out[i] = v
			}
		}
		return tensor.New(tensor.WithBacking(out), tensor.WithShape(X.Shape()...)), nil
	default:
		return nil, ErrCast
	}
}
