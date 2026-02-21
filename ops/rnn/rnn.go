package rnn

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var rnnTypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Int32},
	{tensor.Float32, tensor.Float64},
}

const (
	MinRNNInputs = 3
	MaxRNNInputs = 6
)

// RNN represents the ONNX rnn operator.
type RNN struct {
	ops.BaseOperator

	activationAlpha []float32
	activationBeta  []float32
	activations     []string
	direction       ops.SequenceProcessDirection
	hiddenSize      int
}

// newRNN creates a new rnn operator.
func newRNN(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &RNN{
		BaseOperator: ops.NewBaseOperator(
			version,
			MinRNNInputs,
			MaxRNNInputs,
			typeConstraints,
			"rnn",
		),
		activations: []string{"tanh"},
		direction:   ops.Forward,
	}
}

// Init initializes the rnn operator.
func (r *RNN) Init(n *onnx.NodeProto) error {
	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case ops.ActivationAlphaAttr:
			r.activationAlpha = attr.GetFloats()
		case ops.ActivationBetaAttr:
			r.activationBeta = attr.GetFloats()
		case ops.ActivationsAttr:
			activations := []string{}
			for _, activation := range attr.GetStrings() {
				activations = append(activations, string(activation))
			}

			r.activations = activations
		case ops.ClipAttr:
			return ops.ErrUnsupportedAttribute(attr.GetName(), r)
		case ops.DirectionAttr:
			r.direction = ops.SequenceProcessDirection(attr.GetS())
			if r.direction != ops.Forward {
				return ops.ErrUnsupportedAttribute(attr.GetName(), r)
			}
		case ops.HiddenSizeAttr:
			r.hiddenSize = int(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), r)
		}
	}

	return nil
}

// Apply applies the rnn operator.
func (r *RNN) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	if inputs[4] != nil {
		return nil, ops.ErrUnsupportedInput("sequence lens", r.BaseOperator)
	}

	X := inputs[0]
	seqLength := X.Shape()[0]
	batchSize := X.Shape()[1]

	Wi, err := r.getWeights(inputs[1])
	if err != nil {
		return nil, err
	}

	Ri, err := r.getWeights(inputs[2])
	if err != nil {
		return nil, err
	}

	B := inputs[3]
	if B == nil {
		// 2 is the number of bias matrices required by ONNX definition.
		nBiasMatrices := 2
		B = ops.ZeroTensor(1, nBiasMatrices*r.hiddenSize)
	}

	Wbi, Rbi, err := r.getBiases(B)
	if err != nil {
		return nil, err
	}

	Ht := inputs[5]
	if Ht == nil {
		Ht = ops.ZeroTensor(1, batchSize, r.hiddenSize)
	}

	// Reshape the hidden tensor without the bidirectional dimension, as
	// we do not support bidirectional RNN yet. This is the dimension at
	// index 0.
	if err = Ht.Reshape(Ht.Shape().Clone()[1:]...); err != nil {
		return nil, err
	}

	activation, err := ops.GetActivation(r.activations[0])
	if err != nil {
		return nil, err
	}

	// Pre-transpose weight matrices once (transB=1).
	Wit, err := tensor.Transpose(Wi)
	if err != nil {
		return nil, err
	}

	Rit, err := tensor.Transpose(Ri)
	if err != nil {
		return nil, err
	}

	// Combine W+R biases and expand to (batch, hidden) once before the loop.
	bias, err := tensor.Add(Wbi, Rbi)
	if err != nil {
		return nil, err
	}

	bias, err = expandBias(bias, batchSize)
	if err != nil {
		return nil, err
	}

	inputSize := X.Shape()[2]
	outputs := []tensor.Tensor{}

	// Loop over all timesteps of the input, applying the RNN calculation to every
	// timesteps while updating the hidden tensor.
	for t := range seqLength {
		Xt, err := X.Slice(ops.NewSlicer(t, t+1), nil, nil)
		if err != nil {
			return nil, err
		}

		// Reshape from (1, batch, input) to (batch, input) so MatMul produces 2D results.
		if err = Xt.Reshape(batchSize, inputSize); err != nil {
			return nil, err
		}

		Ht, err = r.layerCalcDirect(Xt, Ht, Wit, Rit, bias, activation)
		if err != nil {
			return nil, err
		}

		outputs = append(outputs, Ht)
	}

	Y := outputs[0]
	if len(outputs) > 1 {
		Y, err = tensor.Concat(0, Y, outputs[1:]...)
		if err != nil {
			return nil, err
		}
	}

	Yh, ok := Ht.Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", Ht.Clone())
	}

	// Reshape the hidden tensor without the bidirectional dimension, as
	// we do not support bidirectional RNN yet. This is the dimension at
	// index 0.
	if err = Y.Reshape(seqLength, 1, batchSize, r.hiddenSize); err != nil {
		return nil, err
	}

	if err = Yh.Reshape(1, batchSize, r.hiddenSize); err != nil {
		return nil, err
	}

	return []tensor.Tensor{Y, Yh}, nil
}

// layerCalcDirect computes the RNN layer using pre-transposed weights and combined bias.
// Ht = activation(Xt @ Wi^T + Ht-1 @ Ri^T + bias)
func (r *RNN) layerCalcDirect(
	Xt, H, Wit, Rit, bias tensor.Tensor, activation ops.Activation,
) (tensor.Tensor, error) {
	inputCalc, err := tensor.MatMul(Xt, Wit)
	if err != nil {
		return nil, err
	}

	hiddenCalc, err := tensor.MatMul(H, Rit)
	if err != nil {
		return nil, err
	}

	sum, err := tensor.Add(inputCalc, hiddenCalc)
	if err != nil {
		return nil, err
	}

	result, err := tensor.Add(sum, bias)
	if err != nil {
		return nil, err
	}

	return activation(result)
}

// expandBias reshapes a 1D bias (hidden) to (1, hidden) and repeats to (batchSize, hidden).
func expandBias(bias tensor.Tensor, batchSize int) (tensor.Tensor, error) {
	hidden := bias.Shape()[0]
	if err := bias.Reshape(1, hidden); err != nil {
		return nil, err
	}

	if batchSize > 1 {
		return tensor.Repeat(bias, 0, batchSize)
	}

	return bias, nil
}

// getWeights returns the weights from a concatenated weight tensor. The result is
// a single weight matrix. W has shape (num_directions, hidden_size, ...).
// The W tensor, by GONNX definition, has 3 dimensions with 1 weight
// tensor in it (2 if bidirectional, but that is not supported).
func (r *RNN) getWeights(W tensor.Tensor) (tensor.Tensor, error) {
	nWeightMatrices := 1
	nWeightDimensions := 3

	weights, err := ops.ExtractMatrices(W, nWeightMatrices, nWeightDimensions, r.hiddenSize)
	if err != nil {
		return nil, err
	}

	return weights[0], nil
}

// getBiases splits tensor B into 2 bias matrices.
// The B tensor, by GONNX definition, has 2 dimensions with 2 bias
// tensors in it (4 if bidirectional, but that is not supported).
func (r *RNN) getBiases(B tensor.Tensor) (Wbi, Rbi tensor.Tensor, err error) {
	nBiasMatrices := 2
	nBiasDimensions := 2

	b, err := ops.ExtractMatrices(B, nBiasMatrices, nBiasDimensions, r.hiddenSize)
	if err != nil {
		return nil, nil, err
	}

	return b[0], b[1], nil
}
