package hmm

import (
	"bitbucket.org/akualab/gjoa/model"
	"bitbucket.org/akualab/gjoa/model/gaussian"
	"bitbucket.org/akualab/gjoa/model/hmm"
	"code.google.com/p/biogo.matrix"
	//	"math/rand"
	"testing"
)

// Tests

func TestGaussian(t *testing.T) {

	// Gaussian 1.
	mean1, em1 := matrix.NewDense([][]float64{{1}})
	if em1 != nil {
		t.Fatal(em1)
	}
	var1, ev1 := matrix.NewDense([][]float64{{1}})
	if ev1 != nil {
		t.Fatal(ev1)
	}
	g1, eg1 := gaussian.NewGaussian(1, mean1, var1, true, true, "g1")
	if eg1 != nil {
		t.Fatal(eg1)
	}

	// Gaussian 2.
	mean2, em2 := matrix.NewDense([][]float64{{4}})
	if em2 != nil {
		t.Fatal(em2)
	}
	var2, ev2 := matrix.NewDense([][]float64{{4}})
	if ev2 != nil {
		t.Fatal(ev2)
	}
	g2, eg2 := gaussian.NewGaussian(1, mean1, var1, true, true, "g2")
	if eg2 != nil {
		t.Fatal(eg2)
	}

	initialStateProbs, esp := matrix.NewDense([][]float64{{0.8}, {0.2}})
	if esp != nil {
		t.Fatal(esp)
	}

	transProbs, etp := matrix.NewDense([][]float64{{0.9, 0.1}, {0.3, 0.7}})
	if etp != nil {
		t.Fatal(etp)
	}

	hmm, e := hmm.NewHMM(transProbs, initialStateProbs, []*gaussian.Gaussian{g1, g2})

	//	p := g.LogProb(obs)
	//t.Logf("Gaussian: %+v", g)
	//	t.Logf("LogProb: %f", p)
	//	t.Logf("Prob: %f", g.Prob(obs))
	// -3.3818

	//	expected := -3.3818
	//	if !model.Comparef64(expected, p) {
	//		t.Errorf("Wrong LogProb. Expected: [%f], Got: [%f]", expected, p)
	//	}
}
