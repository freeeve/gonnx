package gru

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var gruTypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Int32},
	{tensor.Float32, tensor.Float64},
}

const (
	MinGRUInputs = 3
	MaxGRUInputs = 6
)

// GRU represents the ONNX gru operator. It only supports a simple forward gru
// operation with default activations.
type GRU struct {
	ops.BaseOperator

	activationAlpha   []float32
	activationBeta    []float32
	activations       []string
	direction         ops.SequenceProcessDirection
	hiddenSize        int
	linearBeforeReset bool
}

// newGRU creates a new gru operator.
func newGRU(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &GRU{
		BaseOperator: ops.NewBaseOperator(
			version,
			MinGRUInputs,
			MaxGRUInputs,
			typeConstraints,
			"gru",
		),
		activations:       []string{"sigmoid", "tanh"},
		direction:         ops.Forward,
		linearBeforeReset: false,
	}
}

// Init initializes the gru operator. Currently, our GRU operator does not support all
// attributes as specified by the ONNX operator. The basic functionality is working and
// the other attributes can be added later on.
func (g *GRU) Init(n *onnx.NodeProto) error {
	attributes := n.GetAttribute()
	for _, attr := range attributes {
		switch attr.GetName() {
		case ops.ActivationAlphaAttr:
			g.activationAlpha = attr.GetFloats()
		case ops.ActivationBetaAttr:
			g.activationBeta = attr.GetFloats()
		case ops.ActivationsAttr:
			activations := []string{}
			for _, activation := range attr.GetStrings() {
				activations = append(activations, string(activation))
			}

			g.activations = activations
		case ops.ClipAttr:
			return ops.ErrUnsupportedAttribute(attr.GetName(), g)
		case ops.DirectionAttr:
			g.direction = ops.SequenceProcessDirection(attr.GetS())
			if g.direction != ops.Forward {
				return ops.ErrUnsupportedAttribute(attr.GetName(), g)
			}
		case ops.HiddenSizeAttr:
			g.hiddenSize = int(attr.GetI())
		case "linear_before_reset":
			g.linearBeforeReset = ops.Int64ToBool(attr.GetI())
		default:
			return ops.ErrInvalidAttribute(attr.GetName(), g)
		}
	}

	return nil
}

// Apply applies the gru operator.
func (g *GRU) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	if inputs[4] != nil {
		return nil, ops.ErrUnsupportedInput("sequence lens", g.BaseOperator)
	}

	X := inputs[0]
	seqLength := X.Shape()[0]
	batchSize := X.Shape()[1]

	Wz, Wr, Wh, err := g.getWeights(inputs[1])
	if err != nil {
		return nil, err
	}

	Rz, Rr, Rh, err := g.getWeights(inputs[2])
	if err != nil {
		return nil, err
	}

	B := inputs[3]
	if B == nil {
		// 6 is the number of bias matrices required by ONNX definition.
		nBiasMatrices := 6
		B = ops.ZeroTensor(1, nBiasMatrices*g.hiddenSize)
	}

	Wbz, Wbr, Wbh, Rbz, Rbr, Rbh, err := g.getBiases(B)
	if err != nil {
		return nil, err
	}

	prevH := inputs[5]
	if prevH == nil {
		prevH = ops.ZeroTensor(1, batchSize, g.hiddenSize)
	}

	// Extract the shape of the hidden dimensions without the bidirectional dimension, as
	// we do not support bidirectional GRU yet.
	shapeWithoutBidir := prevH.Shape().Clone()[1:]

	err = prevH.Reshape(shapeWithoutBidir...)
	if err != nil {
		return nil, err
	}

	fActivation, err := ops.GetActivation(g.activations[0])
	if err != nil {
		return nil, err
	}

	gActivation, err := ops.GetActivation(g.activations[1])
	if gActivation == nil {
		return nil, err
	}

	// Pre-transpose weight matrices once (all gates use transB=1).
	Wzt, err := tensor.Transpose(Wz)
	if err != nil {
		return nil, err
	}

	Wrt, err := tensor.Transpose(Wr)
	if err != nil {
		return nil, err
	}

	Wht, err := tensor.Transpose(Wh)
	if err != nil {
		return nil, err
	}

	Rzt, err := tensor.Transpose(Rz)
	if err != nil {
		return nil, err
	}

	Rrt, err := tensor.Transpose(Rr)
	if err != nil {
		return nil, err
	}

	Rht, err := tensor.Transpose(Rh)
	if err != nil {
		return nil, err
	}

	// Combine W+R biases for z and r gates (done once before loop).
	biasZ, err := tensor.Add(Wbz, Rbz)
	if err != nil {
		return nil, err
	}

	biasR, err := tensor.Add(Wbr, Rbr)
	if err != nil {
		return nil, err
	}

	// Expand all biases from (hidden) to (batch, hidden) to match MatMul output shapes.
	// Done once before the loop to avoid per-timestep broadcast allocations.
	biasZ, err = ops.ExpandBias(biasZ, batchSize)
	if err != nil {
		return nil, err
	}

	biasR, err = ops.ExpandBias(biasR, batchSize)
	if err != nil {
		return nil, err
	}

	Wbh, err = ops.ExpandBias(Wbh, batchSize)
	if err != nil {
		return nil, err
	}

	Rbh, err = ops.ExpandBias(Rbh, batchSize)
	if err != nil {
		return nil, err
	}

	inputSize := X.Shape()[2]
	outputs := []tensor.Tensor{}

	// Pre-allocate scratch buffers for reuse across timesteps.
	s1 := tensor.New(tensor.WithShape(batchSize, g.hiddenSize), tensor.Of(X.Dtype()))
	s2 := tensor.New(tensor.WithShape(batchSize, g.hiddenSize), tensor.Of(X.Dtype()))
	s3 := tensor.New(tensor.WithShape(batchSize, g.hiddenSize), tensor.Of(X.Dtype()))

	for i := range seqLength {
		Xt, err := g.extractXt(X, i)
		if err != nil {
			return nil, err
		}

		// Reshape from (1, batch, input) to (batch, input) so MatMul produces 2D results.
		if err = Xt.Reshape(batchSize, inputSize); err != nil {
			return nil, err
		}

		zt, err := g.gateCalcDirect(Xt, prevH, Wzt, Rzt, biasZ, s1, s2, fActivation)
		if err != nil {
			return nil, err
		}

		rt, err := g.gateCalcDirect(Xt, prevH, Wrt, Rrt, biasR, s1, s2, fActivation)
		if err != nil {
			return nil, err
		}

		ht, err := g.htCalcDirect(Xt, prevH, rt, Wht, Rht, Wbh, Rbh, s1, s2, s3, gActivation)
		if err != nil {
			return nil, err
		}

		prevH, err = g.hiddenCalcDirect(zt, ht, prevH)
		if err != nil {
			return nil, err
		}

		outputs = append(outputs, prevH)
	}

	var Y tensor.Tensor
	if len(outputs) > 1 {
		Y, err = tensor.Concat(0, outputs[0], outputs[1:]...)
		if err != nil {
			return nil, err
		}
	} else {
		Y = outputs[0]
	}

	// Reshape the output so it adds the num_directions as specified by onnx.
	err = Y.Reshape([]int{seqLength, 1, batchSize, g.hiddenSize}...)
	if err != nil {
		return nil, err
	}

	Yh, ok := prevH.Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", prevH.Clone())
	}

	// Reshape the output so it adds the num_directions as specified by onnx.
	err = Yh.Reshape([]int{1, batchSize, g.hiddenSize}...)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{Y, Yh}, nil
}

// extractXt extracts the value of x for timestep t.
func (g *GRU) extractXt(X tensor.Tensor, t int) (tensor.Tensor, error) {
	return X.Slice(ops.NewSlicer(t, t+1), nil, nil)
}

// gateCalcDirect computes a gate using pre-transposed weights and combined bias.
// gate = activation(Xt @ Wt + H @ Rt + combinedBias)
// s1 and s2 are pre-allocated scratch buffers to avoid per-timestep allocations.
func (g *GRU) gateCalcDirect(
	Xt, H, Wt, Rt, combinedBias, s1, s2 tensor.Tensor, activation ops.Activation,
) (tensor.Tensor, error) {
	_, err := tensor.MatMul(Xt, Wt, tensor.WithReuse(s1))
	if err != nil {
		return nil, err
	}

	_, err = tensor.MatMul(H, Rt, tensor.WithReuse(s2))
	if err != nil {
		return nil, err
	}

	_, err = tensor.Add(s1, s2, tensor.UseUnsafe())
	if err != nil {
		return nil, err
	}

	_, err = tensor.Add(s1, combinedBias, tensor.UseUnsafe())
	if err != nil {
		return nil, err
	}

	return activation(s1)
}

// htCalcDirect computes the ht gate using pre-transposed weights.
// For linearBeforeReset=false: ht = g(Xt @ Wh^T + Wbh + (rt * prevH) @ Rh^T + Rbh)
// For linearBeforeReset=true:  ht = g(Xt @ Wh^T + Wbh + rt * (prevH @ Rh^T + Rbh))
// s1, s2, s3 are pre-allocated scratch buffers to avoid per-timestep allocations.
func (g *GRU) htCalcDirect(
	Xt, prevH, rt, Wht, Rht, Wbh, Rbh, s1, s2, s3 tensor.Tensor, activation ops.Activation,
) (tensor.Tensor, error) {
	if !g.linearBeforeReset {
		_, err := tensor.Mul(rt, prevH, tensor.WithReuse(s3))
		if err != nil {
			return nil, err
		}

		_, err = tensor.MatMul(Xt, Wht, tensor.WithReuse(s1))
		if err != nil {
			return nil, err
		}

		_, err = tensor.MatMul(s3, Rht, tensor.WithReuse(s2))
		if err != nil {
			return nil, err
		}

		_, err = tensor.Add(s1, s2, tensor.UseUnsafe())
		if err != nil {
			return nil, err
		}

		_, err = tensor.Add(s1, Wbh, tensor.UseUnsafe())
		if err != nil {
			return nil, err
		}

		_, err = tensor.Add(s1, Rbh, tensor.UseUnsafe())
		if err != nil {
			return nil, err
		}

		return activation(s1)
	}

	// linearBeforeReset=true path:
	// ht = g(Xt @ Wh^T + Wbh + rt * (prevH @ Rh^T + Rbh))
	_, err := tensor.MatMul(Xt, Wht, tensor.WithReuse(s1))
	if err != nil {
		return nil, err
	}

	_, err = tensor.Add(s1, Wbh, tensor.UseUnsafe())
	if err != nil {
		return nil, err
	}

	_, err = tensor.MatMul(prevH, Rht, tensor.WithReuse(s2))
	if err != nil {
		return nil, err
	}

	_, err = tensor.Add(s2, Rbh, tensor.UseUnsafe())
	if err != nil {
		return nil, err
	}

	_, err = tensor.Mul(s2, rt, tensor.UseUnsafe())
	if err != nil {
		return nil, err
	}

	_, err = tensor.Add(s1, s2, tensor.UseUnsafe())
	if err != nil {
		return nil, err
	}

	return activation(s1)
}

// hiddenCalcDirect computes Ht = (1 - zt) * ht + zt * prevH using direct backing array access.
func (g *GRU) hiddenCalcDirect(zt, ht, prevH tensor.Tensor) (tensor.Tensor, error) {
	out := tensor.New(tensor.WithShape(zt.Shape()...), tensor.Of(zt.Dtype()))

	switch zt.Dtype() {
	case tensor.Float32:
		gruHiddenTyped(out.Data().([]float32), zt.Data().([]float32), ht.Data().([]float32), prevH.Data().([]float32))
	case tensor.Float64:
		gruHiddenTyped(out.Data().([]float64), zt.Data().([]float64), ht.Data().([]float64), prevH.Data().([]float64))
	}

	return out, nil
}

func gruHiddenTyped[T ops.FloatType](out, zt, ht, prevH []T) {
	for i := range out {
		out[i] = (1-zt[i])*ht[i] + zt[i]*prevH[i]
	}
}

// expandBias reshapes a 1D bias (hidden) to (1, hidden) and repeats to (batchSize, hidden).
// getWeights splits tensor W into 3 weight matrices.
// The W tensor, by GONNX definition, has 3 dimensions with 3 weight
// tensors in it (6 if bidirectional, but that is not supported).
func (g *GRU) getWeights(W tensor.Tensor) (Wz, Wr, Wh tensor.Tensor, err error) {
	nWeightMatrices := 3
	nWeightDimensions := 3

	weights, err := ops.ExtractMatrices(W, nWeightMatrices, nWeightDimensions, g.hiddenSize)
	if err != nil {
		return nil, nil, nil, err
	}

	return weights[0], weights[1], weights[2], nil
}

// getBiases returns the biases from the Bias node as specified by the ONNX standard.
// The B tensor, by GONNX definition, has 2 dimensions with 6 bias
// tensors in it (12 if bidirectional, but that is not supported).
func (g *GRU) getBiases(B tensor.Tensor) (Wbz, Wbr, Wbh, Rbz, Rbr, Rbh tensor.Tensor, err error) {
	nBiasMatrices := 6
	nBiasDimensions := 2

	biases, err := ops.ExtractMatrices(B, nBiasMatrices, nBiasDimensions, g.hiddenSize)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}

	return biases[0], biases[1], biases[2], biases[3], biases[4], biases[5], nil
}
