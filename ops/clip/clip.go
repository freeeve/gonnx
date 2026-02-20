package clip

import (
	"math"

	"github.com/advancedclimatesystems/gonnx/onnx"
	"github.com/advancedclimatesystems/gonnx/ops"
	"gorgonia.org/tensor"
)

var clipTypeConstraints = [][]tensor.Dtype{
	{tensor.Float32, tensor.Float64, tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64, tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64},
	{tensor.Float32, tensor.Float64, tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64, tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64},
	{tensor.Float32, tensor.Float64, tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64, tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64},
}

// Clip represents the ONNX clip operator.
type Clip struct {
	ops.BaseOperator
}

// newClip creates a new clip operator.
func newClip(version int, typeConstraints [][]tensor.Dtype) ops.Operator {
	return &Clip{
		BaseOperator: ops.NewBaseOperator(
			version,
			1,
			3,
			typeConstraints,
			"clip",
		),
	}
}

// Init initializes the clip operator.
func (c *Clip) Init(*onnx.NodeProto) error {
	return nil
}

// Apply applies the clip operator.
func (c *Clip) Apply(inputs []tensor.Tensor) ([]tensor.Tensor, error) {
	input := inputs[0]

	minVal := -math.MaxFloat64
	maxVal := math.MaxFloat64

	if len(inputs) > 1 && inputs[1] != nil {
		v, err := getScalarFloat64(inputs[1])
		if err != nil {
			return nil, err
		}
		minVal = v
	}

	if len(inputs) > 2 && inputs[2] != nil {
		v, err := getScalarFloat64(inputs[2])
		if err != nil {
			return nil, err
		}
		maxVal = v
	}

	out := tensor.New(tensor.WithShape(input.Shape()...), tensor.Of(input.Dtype()))

	switch input.Dtype() {
	case tensor.Float32:
		err := clipTyped[float32](out.Data(), input.Data(), float32(minVal), float32(maxVal))
		if err != nil {
			return nil, err
		}
	case tensor.Float64:
		err := clipTyped[float64](out.Data(), input.Data(), minVal, maxVal)
		if err != nil {
			return nil, err
		}
	case tensor.Int8:
		err := clipTyped[int8](out.Data(), input.Data(), int8(clampFloat64(minVal, math.MinInt8, math.MaxInt8)), int8(clampFloat64(maxVal, math.MinInt8, math.MaxInt8)))
		if err != nil {
			return nil, err
		}
	case tensor.Int16:
		err := clipTyped[int16](out.Data(), input.Data(), int16(clampFloat64(minVal, math.MinInt16, math.MaxInt16)), int16(clampFloat64(maxVal, math.MinInt16, math.MaxInt16)))
		if err != nil {
			return nil, err
		}
	case tensor.Int32:
		err := clipTyped[int32](out.Data(), input.Data(), int32(clampFloat64(minVal, math.MinInt32, math.MaxInt32)), int32(clampFloat64(maxVal, math.MinInt32, math.MaxInt32)))
		if err != nil {
			return nil, err
		}
	case tensor.Int64:
		err := clipTyped[int64](out.Data(), input.Data(), int64(clampFloat64(minVal, math.MinInt64, math.MaxInt64)), int64(clampFloat64(maxVal, math.MinInt64, math.MaxInt64)))
		if err != nil {
			return nil, err
		}
	case tensor.Uint8:
		err := clipTyped[uint8](out.Data(), input.Data(), uint8(clampFloat64(minVal, 0, math.MaxUint8)), uint8(clampFloat64(maxVal, 0, math.MaxUint8)))
		if err != nil {
			return nil, err
		}
	case tensor.Uint16:
		err := clipTyped[uint16](out.Data(), input.Data(), uint16(clampFloat64(minVal, 0, math.MaxUint16)), uint16(clampFloat64(maxVal, 0, math.MaxUint16)))
		if err != nil {
			return nil, err
		}
	case tensor.Uint32:
		err := clipTyped[uint32](out.Data(), input.Data(), uint32(clampFloat64(minVal, 0, math.MaxUint32)), uint32(clampFloat64(maxVal, 0, math.MaxUint32)))
		if err != nil {
			return nil, err
		}
	case tensor.Uint64:
		err := clipTyped[uint64](out.Data(), input.Data(), uint64(clampFloat64(minVal, 0, math.MaxUint64)), uint64(clampFloat64(maxVal, 0, math.MaxUint64)))
		if err != nil {
			return nil, err
		}
	default:
		return nil, ops.ErrInvalidInputType(0, input.Dtype().String(), c.BaseOperator)
	}

	return []tensor.Tensor{out}, nil
}

func clipTyped[T float32 | float64 | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64](result, input any, minVal, maxVal T) error {
	res, ok := result.([]T)
	if !ok {
		return ops.ErrTypeAssert("numeric list", result)
	}

	in, ok := input.([]T)
	if !ok {
		return ops.ErrTypeAssert("numeric list", input)
	}

	for i, v := range in {
		if v < minVal {
			v = minVal
		}
		if v > maxVal {
			v = maxVal
		}
		res[i] = v
	}

	return nil
}

func getScalarFloat64(t tensor.Tensor) (float64, error) {
	switch t.Dtype() {
	case tensor.Float32:
		data := t.Data()
		switch d := data.(type) {
		case []float32:
			if len(d) > 0 {
				return float64(d[0]), nil
			}
		case float32:
			return float64(d), nil
		}
	case tensor.Float64:
		data := t.Data()
		switch d := data.(type) {
		case []float64:
			if len(d) > 0 {
				return d[0], nil
			}
		case float64:
			return d, nil
		}
	case tensor.Int8:
		data := t.Data()
		switch d := data.(type) {
		case []int8:
			if len(d) > 0 {
				return float64(d[0]), nil
			}
		case int8:
			return float64(d), nil
		}
	case tensor.Int16:
		data := t.Data()
		switch d := data.(type) {
		case []int16:
			if len(d) > 0 {
				return float64(d[0]), nil
			}
		case int16:
			return float64(d), nil
		}
	case tensor.Int32:
		data := t.Data()
		switch d := data.(type) {
		case []int32:
			if len(d) > 0 {
				return float64(d[0]), nil
			}
		case int32:
			return float64(d), nil
		}
	case tensor.Int64:
		data := t.Data()
		switch d := data.(type) {
		case []int64:
			if len(d) > 0 {
				return float64(d[0]), nil
			}
		case int64:
			return float64(d), nil
		}
	case tensor.Uint8:
		data := t.Data()
		switch d := data.(type) {
		case []uint8:
			if len(d) > 0 {
				return float64(d[0]), nil
			}
		case uint8:
			return float64(d), nil
		}
	case tensor.Uint16:
		data := t.Data()
		switch d := data.(type) {
		case []uint16:
			if len(d) > 0 {
				return float64(d[0]), nil
			}
		case uint16:
			return float64(d), nil
		}
	case tensor.Uint32:
		data := t.Data()
		switch d := data.(type) {
		case []uint32:
			if len(d) > 0 {
				return float64(d[0]), nil
			}
		case uint32:
			return float64(d), nil
		}
	case tensor.Uint64:
		data := t.Data()
		switch d := data.(type) {
		case []uint64:
			if len(d) > 0 {
				return float64(d[0]), nil
			}
		case uint64:
			return float64(d), nil
		}
	}

	return 0, ops.ErrCast
}

// clampFloat64 clamps v to the range [lo, hi] to prevent overflow when converting to integer types.
func clampFloat64(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
