package where

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestWhereInit(t *testing.T) {
	op := whereVersions[9]()
	err := op.Init(nil)
	assert.Nil(t, err)
}

func TestWhere(t *testing.T) {
	tests := []struct {
		version         int64
		condition       []bool
		conditionShape  []int
		backing1        []float32
		backing1Shape   []int
		backing2        []float32
		backing2Shape   []int
		expectedBacking []float32
	}{
		{
			9,
			[]bool{true, false, true},
			[]int{3},
			[]float32{1, 2, 3},
			[]int{3},
			[]float32{4, 5, 6},
			[]int{3},
			[]float32{1, 5, 3},
		},
		{
			9,
			[]bool{true, false, true, false},
			[]int{2, 2},
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]float32{4, 5},
			[]int{1, 2},
			[]float32{1, 5, 3, 5},
		},
		{
			9,
			[]bool{false, true},
			[]int{2},
			[]float32{1, 2, 3, 4},
			[]int{2, 2},
			[]float32{4, 5},
			[]int{1, 2},
			[]float32{4, 2, 4, 4},
		},
		{
			9,
			[]bool{false, false, false, true, true, true},
			[]int{2, 3},
			[]float32{1, 2, 3, 4, 5, 6},
			[]int{2, 3},
			[]float32{4, 5, 6},
			[]int{3},
			[]float32{4, 5, 6, 4, 5, 6},
		},
		{
			9,
			[]bool{false, true, true, false, false, true},
			[]int{2, 3},
			[]float32{1, 2, 3, 4, 5, 6},
			[]int{2, 3},
			[]float32{4, 5, 6},
			[]int{3},
			[]float32{4, 2, 3, 4, 5, 6},
		},
	}

	for _, test := range tests {
		inputs := []tensor.Tensor{
			tensor.New(tensor.WithShape(test.conditionShape...), tensor.WithBacking(test.condition)),
			tensor.New(tensor.WithShape(test.backing1Shape...), tensor.WithBacking(test.backing1)),
			tensor.New(tensor.WithShape(test.backing2Shape...), tensor.WithBacking(test.backing2)),
		}

		op := whereVersions[test.version]()

		res, err := op.Apply(inputs)
		assert.Nil(t, err)
		assert.Equal(t, test.expectedBacking, res[0].Data())
	}
}

func BenchmarkWhere(b *testing.B) {
	n := 10000
	cond := make([]bool, n)
	x := make([]float32, n)
	y := make([]float32, n)

	for i := 0; i < n; i++ {
		cond[i] = rand.Intn(2) == 1
		x[i] = rand.Float32()
		y[i] = rand.Float32()
	}

	op := whereVersions[9]()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		inputs := []tensor.Tensor{
			tensor.New(tensor.WithShape(n), tensor.WithBacking(cond)),
			tensor.New(tensor.WithShape(n), tensor.WithBacking(x)),
			tensor.New(tensor.WithShape(n), tensor.WithBacking(y)),
		}

		_, err := op.Apply(inputs)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWhereBroadcast(b *testing.B) {
	rows, cols := 100, 100
	n := rows * cols
	cond := make([]bool, cols)
	x := make([]float32, n)
	y := make([]float32, n)

	for i := 0; i < cols; i++ {
		cond[i] = rand.Intn(2) == 1
	}

	for i := 0; i < n; i++ {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
	}

	op := whereVersions[9]()

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		inputs := []tensor.Tensor{
			tensor.New(tensor.WithShape(cols), tensor.WithBacking(cond)),
			tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(x)),
			tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(y)),
		}

		_, err := op.Apply(inputs)
		if err != nil {
			b.Fatal(err)
		}
	}
}
