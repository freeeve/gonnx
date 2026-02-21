package ops

import (
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

// Tanh performs the tanh operation on a tensor.
func Tanh(X tensor.Tensor) (tensor.Tensor, error) {
	return tensor.Tanh(X)
}

// Sigmoid performs the sigmoid operation on a tensor: 1 / (1 + exp(-x)).
// Uses in-place operations to minimize intermediate allocations.
func Sigmoid(X tensor.Tensor) (tensor.Tensor, error) {
	// Clone X into a working buffer so we don't modify the input.
	buf, ok := X.Clone().(tensor.Tensor)
	if !ok {
		return nil, ErrTypeAssert("tensor.Tensor", X.Clone())
	}

	// buf = -X (in-place)
	if _, err := tensor.Neg(buf, tensor.UseUnsafe()); err != nil {
		return nil, err
	}

	// buf = exp(-X) (in-place)
	if _, err := tensor.Exp(buf, tensor.UseUnsafe()); err != nil {
		return nil, err
	}

	typedOne, err := GetValueAsTensorType(1.0, buf.Dtype())
	if err != nil {
		return nil, err
	}

	// buf = 1 + exp(-X) (reuse buf)
	if _, err := tensor.Add(typedOne, buf, tensor.WithReuse(buf)); err != nil {
		return nil, err
	}

	// result = 1 / (1 + exp(-X))
	return tensor.Div(typedOne, buf)
}

// ReLU performs the ReLU operation on a tensor using direct backing array manipulation.
func ReLU(X tensor.Tensor) (tensor.Tensor, error) {
	out, ok := X.Clone().(tensor.Tensor)
	if !ok {
		return nil, ErrTypeAssert("tensor.Tensor", X.Clone())
	}

	switch X.Dtype() {
	case tensor.Float32:
		reluTyped(out.Data().([]float32))
	case tensor.Float64:
		reluTyped(out.Data().([]float64))
	default:
		return nil, ErrCast
	}

	return out, nil
}

func reluTyped[T FloatType](d []T) {
	for i, v := range d {
		if v < 0 {
			d[i] = 0
		}
	}
}
