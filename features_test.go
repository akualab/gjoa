package gjoa

import (
	//"fmt"
	"github.com/btracey/sliceops"
	"testing"
)

const (
	EQTOLERANCE = 1E-14
	SMALL       = 10
	MEDIUM      = 1000
	LARGE       = 100000
	HUGE        = 10000000
)

func AreSlicesEqual(t *testing.T, truth, comp []float64, str string) {
	if !sliceops.Eq(comp, truth, EQTOLERANCE) {
		t.Errorf(str+"Expected %v, returned %v", truth, comp)
	}
}

func ConstantSlice(f []float64) ProcessFunc {

	return func(proc *Processor, n uint64) []float64 {
		return f
	}
}

var Adder = func(p *Processor, n uint64) []float64 {

	out := make([]float64, p.NumElements())
	sources := p.Inputs()
	for _, s := range sources {
		sliceops.Add(out, s.Get(n))
	}
	return out
}

func Test1(t *testing.T) {

	p1 := NewProcessor("p1", 3, 10, ConstantSlice([]float64{1.1, 2.2, 3.3}), nil)
	p2 := NewProcessor("p2", 3, 3, ConstantSlice([]float64{3.3, 2.2, 1.1}), nil)
	p3 := NewProcessor("p3", 3, 2, Adder, p1, p2)
	truth := []float64{4.4, 4.4, 4.4}

	AreSlicesEqual(t, truth, p3.Get(0), "Wrong addition.")
	t.Logf("%d: %+v", 0, p3.Get(0))
	AreSlicesEqual(t, truth, p3.Get(1), "Wrong addition.")
	AreSlicesEqual(t, truth, p3.Get(0), "Wrong addition.")
	AreSlicesEqual(t, truth, p3.Get(2), "Wrong addition.")
	AreSlicesEqual(t, truth, p3.Get(3), "Wrong addition.")
	AreSlicesEqual(t, truth, p3.Get(4), "Wrong addition.")
	AreSlicesEqual(t, truth, p3.Get(4), "Wrong addition.")
}
