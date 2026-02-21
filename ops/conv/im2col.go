package conv

import (
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

// im2col2D extracts all convolution patches from a single batch sample into a column matrix.
// paddedX has shape [N, C, H, W]. Returns matrix [outH*outW, C*kH*kW] for the given batch index.
func im2col2D(paddedX tensor.Tensor, batchIdx int, kernelShape, strides []int, outH, outW int) (*tensor.Dense, error) {
	switch paddedX.Dtype() {
	case tensor.Float32:
		return im2col2DTyped[float32](paddedX, batchIdx, kernelShape, strides, outH, outW)
	case tensor.Float64:
		return im2col2DTyped[float64](paddedX, batchIdx, kernelShape, strides, outH, outW)
	default:
		return nil, ops.ErrCast
	}
}

func im2col2DTyped[T ops.FloatType](paddedX tensor.Tensor, batchIdx int, kernelShape, strides []int, outH, outW int) (*tensor.Dense, error) {
	shape := paddedX.Shape()
	C := shape[1]
	pH := shape[2]
	pW := shape[3]
	kH := kernelShape[0]
	kW := kernelShape[1]
	strideH := strides[0]
	strideW := strides[1]

	colRows := outH * outW
	colCols := C * kH * kW

	col := make([]T, colRows*colCols)
	data := paddedX.Data().([]T)

	// Offset into the batch: batchIdx * C * pH * pW
	batchOffset := batchIdx * C * pH * pW

	for oh := 0; oh < outH; oh++ {
		for ow := 0; ow < outW; ow++ {
			row := oh*outW + ow
			colIdx := 0

			for c := 0; c < C; c++ {
				channelOffset := batchOffset + c*pH*pW

				for kh := 0; kh < kH; kh++ {
					ih := oh*strideH + kh
					rowOffset := channelOffset + ih*pW

					for kw := 0; kw < kW; kw++ {
						iw := ow*strideW + kw
						col[row*colCols+colIdx] = data[rowOffset+iw]
						colIdx++
					}
				}
			}
		}
	}

	return tensor.NewDense(paddedX.Dtype(), tensor.Shape{colRows, colCols}, tensor.WithBacking(col)), nil
}

// im2col1D extracts all convolution patches from a single batch sample into a column matrix.
// paddedX has shape [N, C, H]. Returns matrix [outH, C*kH] for the given batch index.
func im2col1D(paddedX tensor.Tensor, batchIdx int, kernelShape, strides []int, outH int) (*tensor.Dense, error) {
	switch paddedX.Dtype() {
	case tensor.Float32:
		return im2col1DTyped[float32](paddedX, batchIdx, kernelShape, strides, outH)
	case tensor.Float64:
		return im2col1DTyped[float64](paddedX, batchIdx, kernelShape, strides, outH)
	default:
		return nil, ops.ErrCast
	}
}

func im2col1DTyped[T ops.FloatType](paddedX tensor.Tensor, batchIdx int, kernelShape, strides []int, outH int) (*tensor.Dense, error) {
	shape := paddedX.Shape()
	C := shape[1]
	pH := shape[2]
	kH := kernelShape[0]
	strideH := strides[0]

	colCols := C * kH

	col := make([]T, outH*colCols)
	data := paddedX.Data().([]T)

	batchOffset := batchIdx * C * pH

	for oh := 0; oh < outH; oh++ {
		colIdx := 0

		for c := 0; c < C; c++ {
			channelOffset := batchOffset + c*pH

			for kh := 0; kh < kH; kh++ {
				ih := oh*strideH + kh
				col[oh*colCols+colIdx] = data[channelOffset+ih]
				colIdx++
			}
		}
	}

	return tensor.NewDense(paddedX.Dtype(), tensor.Shape{outH, colCols}, tensor.WithBacking(col)), nil
}
