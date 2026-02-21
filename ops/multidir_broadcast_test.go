package ops

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func BenchmarkMultidirectionalBroadcast(b *testing.B) {
	benchmarks := []struct {
		name   string
		shapeA []int
		shapeB []int
	}{
		{"SameShape", []int{256, 256}, []int{256, 256}},
		{"ScalarBroadcast", []int{256, 256}, []int{1}},
		{"RowBroadcast", []int{256, 256}, []int{1, 256}},
		{"3D", []int{8, 64, 64}, []int{1, 64, 64}},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			A := Float32TensorFixture(bm.shapeA...)
			B := Float32TensorFixture(bm.shapeB...)

			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_, _, err := MultidirectionalBroadcast(A, B)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestMultidirectionalBroadcast(t *testing.T) {
	tests := []struct {
		shapes        [][]int
		expectedShape tensor.Shape
		err           error
	}{
		{
			[][]int{{2}, {2, 2}},
			[]int{2, 2},
			nil,
		},
		{
			[][]int{{2, 3, 4, 5}, {}},
			[]int{2, 3, 4, 5},
			nil,
		},
		{
			[][]int{{2, 3, 4, 5}, {5}},
			[]int{2, 3, 4, 5},
			nil,
		},
		{
			[][]int{{4, 5}, {2, 3, 4, 5}},
			[]int{2, 3, 4, 5},
			nil,
		},
		{
			[][]int{{1, 4, 5}, {2, 3, 1, 1}},
			[]int{2, 3, 4, 5},
			nil,
		},
		{
			[][]int{{3, 4, 5}, {2, 1, 1, 1}},
			[]int{2, 3, 4, 5},
			nil,
		},
		{
			[][]int{{1, 4, 5}, {2, 1, 1, 3}},
			nil,
			ErrMultidirBroadcast([]int{1, 4, 5}, []int{2, 1, 1, 3}, ErrIncompatibleDimensions()),
		},
		{
			[][]int{{5}, {2, 3, 4}},
			nil,
			ErrMultidirBroadcast([]int{5}, []int{2, 3, 4}, ErrIncompatibleDimensions()),
		},
	}

	for _, test := range tests {
		A := Float32TensorFixture(test.shapes[0]...)
		B := Float32TensorFixture(test.shapes[1]...)

		newA, newB, err := MultidirectionalBroadcast(A, B)

		assert.Equal(t, test.err, err)

		if err == nil {
			assert.Equal(t, test.expectedShape, newA.Shape())
			assert.Equal(t, test.expectedShape, newB.Shape())
		} else {
			assert.Nil(t, newA)
			assert.Nil(t, newB)
		}
	}
}
