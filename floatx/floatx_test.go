package floatx

import (
	"github.com/gonum/floats"
	"testing"
)

func TestFlatten2D(t *testing.T) {

	s2d := [][]float64{{11, 22}, {33, 44}, {55, 66}}
	expected := []float64{11, 22, 33, 44, 55, 66}

	flatten := Flatten2D(s2d)
	if !floats.Equal(flatten, expected) {
		t.Fatalf("Flatten failed. expected %+v, got %+v", expected, flatten)
	}
}
