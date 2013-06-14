package model

import (
	//"code.google.com/p/biogo.matrix"
	"testing"
)

// Tests
func TestGaussian(t *testing.T) {

	g, e := NewGaussian(10, nil, nil, true, true, "testing")
	if e != nil {
		t.Fatal(e)
	}
	t.Logf("Gaussian: %+v", g)
}
