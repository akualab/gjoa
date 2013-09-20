package hmm

import (
	"github.com/akualab/gjoa/model"
	"github.com/akualab/gjoa/model/gaussian"
	"math/rand"
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

func MakeRandHMM(t *testing.T, seed int64) *HMM {
	r := rand.New(rand.NewSource(seed))
	mean0 := []float64{3, 3}
	var0 := []float64{0.5, 0.5}
	g0, eg0 := gaussian.NewGaussian(2, mean0, var0, true, true, "g0")
    if eg0 != nil {
        t.Fatal(eg0)
    }
    g1, eg1 := gaussian.NewGaussian(2, mean0, var0, true, true, "g1")
    if eg1 != nil {
        t.Fatal(eg1)
    }
	ran0 := r.Float64()
	ran1 := r.Float64()
	ran2 := r.Float64()
	initialStateProbs := []float64{ran0, 1.0 - ran0}
    transProbs := [][]float64{{ran1, 1- ran1}, {ran2, 1-ran2}}
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

// Still needs work
func TestTrainHMM(t *testing.T) {

	hmm0 := MakeHMM2(t)
	hmm := MakeRandHMM(t, 3)
	// number of updates
	iter := 1
	// size of the generated sequence
	n := 10
	// number of sequences
	m := 100
	for i := 0; i < iter; i++ {
		// fix the seed to get the same sequence
		gen := MakeHMMGenerator(hmm0, 33)
		for j := 0; j < m; j++ {
			obs, states, err := gen.next(n)
			if err != nil {
				t.Fatal(err)
			}
			//hmm.Update(obs, 1.0)
			t.Logf("obs[0] states[0] %v %v", obs[0], states[0])
		}
		//hmm.Estimate()
		// t.Logf here
		// Prepare for next iteration.
        //hmm.Clear()
	}
	//var m0 *gaussian.Gaussian
	m0 := hmm.obsModels[0].(*gaussian.Gaussian)
	t.Logf("mean[0] %v", m0.Mean())
}
