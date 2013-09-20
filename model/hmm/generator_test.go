package hmm

import (
	"github.com/akualab/gjoa/model"
	"github.com/akualab/gjoa/model/gaussian"
	"testing"
)

func MakeHMM2(t *testing.T) *HMM {

	mean0 := []float64{1, 2}
	var0 := []float64{0.09, 0.09}
	mean1 := []float64{4, 4}
	var1 := []float64{1, 1}

	g0, eg0 := gaussian.NewGaussian(2, mean0, var0, true, true, "g0")
	if eg0 != nil {
		t.Fatal(eg0)
	}
	g1, eg1 := gaussian.NewGaussian(2, mean1, var1, true, true, "g1")
	if eg1 != nil {
		t.Fatal(eg1)
	}
	initialStateProbs := []float64{0.8, 0.2}
	transProbs := [][]float64{{0.9, 0.1}, {0.3, 0.7}}
	models := []*gaussian.Gaussian{g0, g1}
	m := make([]model.Trainer, len(models))
	for i, v := range models {
		m[i] = v
	}
	hmm, e := NewHMM(transProbs, initialStateProbs, m, false, "testhmm")
	if e != nil {
		t.Fatal(e)
	}
	return hmm
}

func TestHMMGenerator(t *testing.T) {

	hmm := MakeHMM2(t)
	// size of the generated sequence
	n := 10
	gen := MakeHMMGenerator(hmm, 33)
	obs, states, err := gen.next(n)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("States %v", states)
	t.Logf("Seq %v", obs)
}
