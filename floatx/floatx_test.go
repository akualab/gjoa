package floatx

import (
	"testing"

	"github.com/gonum/floats"
)

func TestFlatten2D(t *testing.T) {

	s2d := [][]float64{{11, 22}, {33, 44}, {55, 66}}
	expected := []float64{11, 22, 33, 44, 55, 66}

	flatten := Flatten2D(s2d)
	if !floats.Equal(flatten, expected) {
		t.Fatalf("Flatten failed. expected %+v, got %+v", expected, flatten)
	}
}

func TestCopy2D(t *testing.T) {

	s1 := [][]float64{{11, 22}, {33, 44}, {55, 66}}

	s2 := CopyFloat2D(s1)

	for k, _ := range s1 {
		if &s1[k] == &s2[k] {
			t.Fatalf("Slices have the same address, not a copy.")
		}
		if !floats.Equal(s1[k], s2[k]) {
			t.Fatalf("Copy failed. want: %+v, have: %+v", s1, s2)
		}
	}
}
