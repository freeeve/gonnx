package conv

import (
	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var convTypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
	{tensor.Float32, tensor.Float64},
}

var (
	MinConvInputs      = 2
	MaxConvInputs      = 3
	NDims1DConvolution = 3
	NDims2DConvolution = 4
)

type AutoPadSetting string

const (
	NotSet    AutoPadSetting = "NOTSET"
	SameUpper AutoPadSetting = "SAME_UPPER"
	SameLower AutoPadSetting = "SAME_LOWER"
	Valid     AutoPadSetting = "VALID"
)

// The number of non spatial dimensions inputs and kernels will always have.
// For input tensors, the first dimension will be the batch size.
// For kernel tensors, the first dimension will be the number of kernels.
// For all tensors, the second dimension will be the number of channels.
const nNonSpatialDims = 2

// Conv represents the ONNX conv operator.
type Conv struct {
	ops.BaseOperator

	autoPad     AutoPadSetting
	dilations   []int
	group       int
	kernelShape []int
	pads        []int
	strides     []int
}

// newConv creates a new conv operator.
func newConv(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Conv{
		BaseOperator: ops.NewBaseOperator(
			version,
			MinConvInputs,
			MaxConvInputs,
			typeConstraints,
			"conv",
		),
		autoPad: NotSet,
	}
}

// Init initializes the conv operator.
func (c *Conv) Init(n *onnx.NodeProto) error {
	var err error

	for _, attr := range n.GetAttribute() {
		switch attr.GetName() {
		case "auto_pad":
			c.autoPad = AutoPadSetting(attr.GetS())
		case "dilations":
			c.dilations, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return ops.ErrInvalidAttribute(attr.GetName(), c)
			}
		case "group":
			c.group = int(attr.GetI())
			if c.group != 1 {
				return ops.ErrUnsupportedAttribute(attr.GetName(), c)
			}
		case "kernel_shape":
			c.kernelShape, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return ops.ErrInvalidAttribute(attr.GetName(), c)
			}
		case "pads":
			c.pads, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return ops.ErrInvalidAttribute(attr.GetName(), c)
			}
		case "strides":
			c.strides, err = ops.AnyToIntSlice(attr.GetInts())
			if err != nil {
				return ops.ErrInvalidAttribute(attr.GetName(), c)
			}
		default:
			return ops.ErrUnsupportedAttribute(attr.GetName(), c)
		}
	}

	return nil
}

// Apply applies the conv operator.
func (c *Conv) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	x := inputs[0]
	kernel := inputs[1]
	bias := inputs[2]

	if len(c.dilations) == 0 {
		c.setDefaultDilations(x)
	}

	if len(c.kernelShape) == 0 {
		c.setKernelShape(kernel)
	}

	if len(c.pads) == 0 {
		c.setDefaultPaddings(x)
	}

	if len(c.strides) == 0 {
		c.setDefaultStrides(x)
	}

	kernel, err := c.getDilatedKernel(kernel)
	if err != nil {
		return nil, err
	}

	if c.autoPad != NotSet {
		c.setPaddingWithAutoPad(x)
	}

	var out tensor.Tensor

	switch len(x.Shape()) {
	case NDims1DConvolution:
		out, err = c.applyConv1D(x, kernel)
	case NDims2DConvolution:
		out, err = c.applyConv2D(x, kernel)
	default:
		return nil, ops.ErrInvalidInput("the convolution operator currently only supports 1D or 2D convolution, i.e. shape [N x C x H (x W)]", c.BaseOperator)
	}

	if err != nil {
		return nil, err
	}

	if bias != nil {
		out, err = c.addBias(out, bias)
		if err != nil {
			return nil, err
		}
	}

	return []tensor.Tensor{out}, nil
}

// setDefaultDilations sets the dilations attribute to the default. Can be called when no
// dilations were set when initializing.
func (c *Conv) setDefaultDilations(x tensor.Tensor) {
	nDims := len(x.Shape()[2:])

	dilations := make([]int, nDims)
	for i := 0; i < nDims; i++ {
		dilations[i] = 1
	}

	c.dilations = dilations
}

// setKernelShape infers the shape of the kernel when it was not given in the attributes.
func (c *Conv) setKernelShape(kernel tensor.Tensor) {
	c.kernelShape = kernel.Shape()[2:]
}

// setDefaultPaddings sets default paddings as attribute. Can be called when no paddings
// were set during initialization.
func (c *Conv) setDefaultPaddings(x tensor.Tensor) {
	NPadsPerDim := 2
	paddingLength := len(x.Shape()[2:]) * NPadsPerDim

	pads := make([]int, paddingLength)
	for i := 0; i < paddingLength; i++ {
		pads[i] = 0
	}

	c.pads = pads
}

// setDefaultStrides sets default strides as attribute. Can be called when no strides
// were set during initialization.
func (c *Conv) setDefaultStrides(x tensor.Tensor) {
	nDims := len(x.Shape()[2:])

	strides := make([]int, nDims)
	for i := 0; i < nDims; i++ {
		strides[i] = 1
	}

	c.strides = strides
}

// setPaddingWithAutoPad sets the padding attribute of the operator based on
// the input tensor `x`, the shape of the kernel and the strides.
func (c *Conv) setPaddingWithAutoPad(x tensor.Tensor) {
	if c.autoPad == NotSet {
		return
	}

	NPadsPerDim := 2
	inputShape := x.Shape()
	nDims := len(inputShape)
	nSpatialDims := nDims - nNonSpatialDims

	c.pads = make([]int, nSpatialDims*NPadsPerDim)

	for i := 0; i < nSpatialDims; i++ {
		dim := inputShape[i]
		targetSize := (dim + c.strides[i] - 1) / c.strides[i]
		padNeeded := (targetSize-1)*c.strides[i] + c.kernelShape[i] - dim

		var padHead int
		if c.autoPad == SameLower {
			// nolint as the division by zero is literally division by two
			padHead = (padNeeded + 1) / 2
		} else {
			// nolint as the division by two is literally division by two
			padHead = padNeeded / 2
		}

		padTail := padNeeded - padHead
		c.pads[i] = padHead
		c.pads[i+nSpatialDims] = padTail
	}
}

// getDilatedKernel creates a new kernel given the `dilations` attribute of this
// conv operator. A dilated kernel basically means inserting zeros in between
// the kernels, i.e. a 2D kernel like:
//
//	1 2
//	3 4
//
// Dilated by one in both dimensions yields a new kernel of:
//
//	1 0 2
//	0 0 0
//	3 0 4
//
// This function updates the given kernel and dilates it by the given amount
// for each dimensions separately. It returns a new tensor with the new kernel.
func (c *Conv) getDilatedKernel(kernel tensor.Tensor) (tensor.Tensor, error) {
	oldKernelShape := kernel.Shape()
	newKernelShape := make([]int, len(oldKernelShape))

	// Add the non spatial dimensions of the kernel, i.e. the number of
	// kernels (index 0) and the number of channels (index 1). These
	// dimensions do not have to be dilated.
	for i := 0; i < nNonSpatialDims; i++ {
		newKernelShape[i] = oldKernelShape[i]
	}

	// Add the dilated spatial dimensions of the kernel, i.e. in the case
	// of 2D images these are the width and height dimensions.
	for i, dilation := range c.dilations {
		oldKernelDim := oldKernelShape[nNonSpatialDims+i]
		newKernelShape[nNonSpatialDims+i] = oldKernelDim + (oldKernelDim-1)*(dilation-1)
	}

	newKernel := tensor.NewDense(kernel.Dtype(), newKernelShape)
	newKernel.Zero()

	// Now we fill the empty kernel with the original kernel values at the
	// right positions.
	iterator := kernel.Iterator()
	iterator.Reset()

	for !iterator.Done() {
		oldCoords := iterator.Coord()

		value, err := kernel.At(oldCoords...)
		if err != nil {
			return nil, err
		}

		newCoords := c.getNewCoordsAfterDilation(oldCoords)

		err = newKernel.SetAt(value, newCoords...)
		if err != nil {
			return nil, err
		}

		_, err = iterator.Next()
		if err != nil {
			return nil, err
		}
	}

	c.setKernelShape(newKernel)

	return newKernel, nil
}

// getNewCoordsAfterDilation returns the new coordinates of a value given the old coordinates of that
// value in the old kernel and its shape. The new coordinates can be used to store the value/weight
// in the dilated kernel.
func (c *Conv) getNewCoordsAfterDilation(oldCoords []int) []int {
	newCoords := make([]int, len(oldCoords))

	for i := 0; i < nNonSpatialDims; i++ {
		newCoords[i] = oldCoords[i]
	}

	for i, dilation := range c.dilations {
		newCoords[nNonSpatialDims+i] = oldCoords[nNonSpatialDims+i] * dilation
	}

	return newCoords
}

// prepareKernelMatrix clones the kernel, broadcasts channels to match inputC if needed,
// reshapes to [M, inputC * prod(kernelShape)], and transposes to [inputC * prod(kernelShape), M].
func (c *Conv) prepareKernelMatrix(kernel tensor.Tensor, inputC int) (tensor.Tensor, error) {
	kData, ok := kernel.Clone().(tensor.Tensor)
	if !ok {
		return nil, ops.ErrTypeAssert("tensor.Tensor", kernel.Clone())
	}

	kernelC := kernel.Shape()[1]
	if kernelC < inputC {
		var err error
		kData, err = tensor.Repeat(kData, 1, inputC/kernelC)
		if err != nil {
			return nil, err
		}
	}

	nKernels := kernel.Shape()[0]
	spatialSize := 1
	for _, ks := range c.kernelShape {
		spatialSize *= ks
	}

	flatCols := inputC * spatialSize
	if err := kData.Reshape(nKernels, flatCols); err != nil {
		return nil, err
	}

	return tensor.Transpose(kData)
}

// applyConv1D applies 1D convolution using im2col + MatMul.
// X has shape [N, C, H], kernel has shape [M, C, kH].
func (c *Conv) applyConv1D(x, kernel tensor.Tensor) (tensor.Tensor, error) {
	outputShape := c.getOutputShape(x, kernel)

	paddedX, err := c.padInput(x)
	if err != nil {
		return nil, err
	}

	nBatches := x.Shape()[0]
	inputC := paddedX.Shape()[1]
	nKernels := kernel.Shape()[0]
	outH := outputShape[nNonSpatialDims]

	// Prepare kernel matrix: broadcast channels if needed, reshape [M, C*kH], then transpose.
	kernelT, err := c.prepareKernelMatrix(kernel, inputC)
	if err != nil {
		return nil, err
	}

	out := tensor.NewDense(x.Dtype(), outputShape)

	for batchIdx := range nBatches {
		colMatrix, err := im2col1D(paddedX, batchIdx, c.kernelShape, c.strides, outH)
		if err != nil {
			return nil, err
		}

		// colMatrix [outH, C*kH] @ kernelT [C*kH, M] -> result [outH, M]
		result, err := tensor.MatMul(colMatrix, kernelT)
		if err != nil {
			return nil, err
		}

		if err := c.copyResultToOutput1D(out, result, batchIdx, nKernels, outH); err != nil {
			return nil, err
		}
	}

	return out, nil
}

// applyConv2D applies 2D convolution using im2col + MatMul.
// X has shape [N, C, H, W], kernel has shape [M, C, kH, kW].
func (c *Conv) applyConv2D(x, kernel tensor.Tensor) (tensor.Tensor, error) {
	outputShape := c.getOutputShape(x, kernel)

	paddedX, err := c.padInput(x)
	if err != nil {
		return nil, err
	}

	nBatches := x.Shape()[0]
	inputC := paddedX.Shape()[1]
	nKernels := kernel.Shape()[0]
	outH := outputShape[nNonSpatialDims]
	outW := outputShape[nNonSpatialDims+1]

	// Prepare kernel matrix: broadcast channels if needed, reshape [M, C*kH*kW], then transpose.
	kernelT, err := c.prepareKernelMatrix(kernel, inputC)
	if err != nil {
		return nil, err
	}

	out := tensor.NewDense(x.Dtype(), outputShape)

	for batchIdx := range nBatches {
		colMatrix, err := im2col2D(paddedX, batchIdx, c.kernelShape, c.strides, outH, outW)
		if err != nil {
			return nil, err
		}

		// colMatrix [outH*outW, C*kH*kW] @ kernelT [C*kH*kW, M] -> result [outH*outW, M]
		result, err := tensor.MatMul(colMatrix, kernelT)
		if err != nil {
			return nil, err
		}

		if err := c.copyResultToOutput2D(out, result, batchIdx, nKernels, outH, outW); err != nil {
			return nil, err
		}
	}

	return out, nil
}

// getOutputShape calculates the shape of the output tensor resulting from
// the convolution operation between `x` and `kernel`.
// `x` has shape [N, C, H, W, ...] and `kernel` has shape [M, C, H, W, ...].
// The output shape will be [N, M, newH, newW, ...], where values like `newH`
// are calculated based on the input shape, kernel size, padding and strides.
func (c *Conv) getOutputShape(x, kernel tensor.Tensor) tensor.Shape {
	outputShape := make([]int, len(x.Shape()))

	outputShape[0] = x.Shape()[0]
	outputShape[1] = kernel.Shape()[0]

	nSpatialDims := len(x.Shape()) - nNonSpatialDims
	for i := 0; i < nSpatialDims; i++ {
		inputDim := x.Shape()[nNonSpatialDims+i]
		kernelDim := c.kernelShape[i]
		outputShape[nNonSpatialDims+i] = ((inputDim - kernelDim + c.pads[i] + c.pads[i+nSpatialDims]) / c.strides[i]) + 1
	}

	return outputShape
}

// padInput pads the input with zeros according to the `pads` attribute.
// The pad attribute specifies how many zeros should be added before and
// after the values in that specific dimension.
// Please note that according to ONNX specs, the `pads` attributes is an
// array with pads as [x1_begin, x2_begin, ..., x1_after, x2_after].
// This method achieves padding by concatting tensors with zero values
// before and after each spatial dimension of the input tensor `x`.
func (c *Conv) padInput(x tensor.Tensor) (tensor.Tensor, error) {
	var err error

	nSpatialDims := len(x.Shape()[nNonSpatialDims:])

	for i := 0; i < nSpatialDims; i++ {
		if c.pads[i] != 0 {
			padsBeforeShape := x.Shape().Clone()
			padsBeforeShape[nNonSpatialDims+i] = c.pads[i]
			zerosBefore := tensor.Tensor(tensor.NewDense(x.Dtype(), padsBeforeShape))
			zerosBefore.Zero()

			x, err = tensor.Concat(nNonSpatialDims+i, zerosBefore, x)
			if err != nil {
				return nil, err
			}
		}

		if c.pads[i+nSpatialDims] != 0 {
			padsAfterShape := x.Shape().Clone()
			padsAfterShape[nNonSpatialDims+i] = c.pads[i+nSpatialDims]
			zerosAfter := tensor.Tensor(tensor.NewDense(x.Dtype(), padsAfterShape))
			zerosAfter.Zero()

			x, err = tensor.Concat(nNonSpatialDims+i, x, zerosAfter)
			if err != nil {
				return nil, err
			}
		}
	}

	return x, nil
}

// copyResultToOutput1D copies a MatMul result [outH, M] into the output tensor [N, M, outH]
// at the given batch index. The result layout is row-major: result[h][m].
// The output layout is [batchIdx, kernelIdx, h].
func (c *Conv) copyResultToOutput1D(out *tensor.Dense, result tensor.Tensor, batchIdx, nKernels, outH int) error {
	switch out.Dtype() {
	case tensor.Float32:
		copyResult1DTyped(out.Data().([]float32), result.Data().([]float32), batchIdx, nKernels, outH)
	case tensor.Float64:
		copyResult1DTyped(out.Data().([]float64), result.Data().([]float64), batchIdx, nKernels, outH)
	default:
		return ops.ErrCast
	}

	return nil
}

func copyResult1DTyped[T ops.FloatType](outData, resData []T, batchIdx, nKernels, outH int) {
	batchOffset := batchIdx * nKernels * outH

	for h := range outH {
		for m := range nKernels {
			outData[batchOffset+m*outH+h] = resData[h*nKernels+m]
		}
	}
}

// copyResultToOutput2D copies a MatMul result [outH*outW, M] into the output tensor [N, M, outH, outW]
// at the given batch index.
func (c *Conv) copyResultToOutput2D(out *tensor.Dense, result tensor.Tensor, batchIdx, nKernels, outH, outW int) error {
	switch out.Dtype() {
	case tensor.Float32:
		copyResult2DTyped(out.Data().([]float32), result.Data().([]float32), batchIdx, nKernels, outH, outW)
	case tensor.Float64:
		copyResult2DTyped(out.Data().([]float64), result.Data().([]float64), batchIdx, nKernels, outH, outW)
	default:
		return ops.ErrCast
	}

	return nil
}

func copyResult2DTyped[T ops.FloatType](outData, resData []T, batchIdx, nKernels, outH, outW int) {
	spatialSize := outH * outW
	batchOffset := batchIdx * nKernels * spatialSize

	for hw := range spatialSize {
		for m := range nKernels {
			outData[batchOffset+m*spatialSize+hw] = resData[hw*nKernels+m]
		}
	}
}

// addBias adds a bias to the output of the convolution. It reshapes the
// bias such that it can be broadcasted, and then is added to the output
// tensor.
func (c *Conv) addBias(out, bias tensor.Tensor) (tensor.Tensor, error) {
	biasShape := make([]int, len(out.Shape()))
	for i := 0; i < len(out.Shape()); i++ {
		biasShape[i] = 1
	}

	biasShape[1] = bias.Shape()[0]

	err := bias.Reshape(biasShape...)
	if err != nil {
		return nil, err
	}

	out, bias, err = ops.UnidirectionalBroadcast(out, bias)
	if err != nil {
		return nil, err
	}

	return tensor.Add(out, bias)
}
