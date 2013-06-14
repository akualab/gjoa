package model

import (
	"code.google.com/p/biogo.matrix"
	"testing"
)

// Tests
func TestGaussian(t *testing.T) {

	g, e := NewGaussian(10, nil, nil, true, true, "testing")
	if e != nil {
		t.Fatal(e)
	}
	t.Logf("Gaussian: %+v", g)

	mean, e2 := matrix.NewDense([][]float64{{0.5}, {1}, {2}})
	if e2 != nil {
		t.Fatal(e2)
	}
	variance, ev := matrix.NewDense([][]float64{{1}, {1}, {1}})
	if ev != nil {
		t.Fatal(ev)
	}

	g, e = NewGaussian(3, mean, variance, true, true, "testing")
	if e != nil {
		t.Fatal(e)
	}
	obs, e3 := matrix.NewDense([][]float64{{1}, {1}, {1}})
	if e3 != nil {
		t.Fatal(e3)
	}
	p := g.LogProb(obs)
	t.Logf("Gaussian: %+v", g)
	t.Logf("LogProb: %f", p)
	// -3.3818
}
