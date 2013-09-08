package hmm

import (
	"bitbucket.org/akualab/gjoa/model"
	"bitbucket.org/akualab/gjoa/model/gaussian"
	"code.google.com/p/biogo.matrix"
	"math"
	"testing"
)

// Tests

// Test ColumnAt() function.
func TestColumnAt(t *testing.T) {

	mat, e := matrix.NewDense([][]float64{{10, 11, 12, 13}, {20, 21, 22, 23}, {30, 31, 32, 33}})
	if e != nil {
		t.Fatal(e)
	}

	col := ColumnAt(mat, 1)

	t.Logf("col: \n%+v", col)

	for i, expected := range []float64{11.0, 21.0, 31.0} {
		v := col.At(i, 0)
		if !model.Comparef64(expected, v) {
			t.Errorf("Wrong value. Expected: [%f], Got: [%f]", expected, v)
		}
	}
}

func TestEvaluation(t *testing.T) {

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
	g2, eg2 := gaussian.NewGaussian(1, mean2, var2, true, true, "g2")
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

	// These are the models.
	models := []*gaussian.Gaussian{g1, g2}

	// To pass the to an HMM we need to convert []*gaussian.Gaussian[] to []model.Modeler
	// see http://golang.org/doc/faq#convert_slice_of_interface
	m := make([]model.Modeler, len(models))
	for i, v := range models {
		m[i] = v
	}

	hmm, e := NewHMM(transProbs, initialStateProbs, m)
	if e != nil {
		t.Fatal(e)
	}

	t.Logf("hmm: %+v", hmm)

	// Gaussian 1.
	obs, eobs := matrix.NewDense([][]float64{{0.1, 0.3, 1.1, 1.2, 0.7, 0.7, 5.5, 7.8, 10.0, 5.2, 1.1, 1.3}})
	if eobs != nil {
		t.Fatal(eobs)
	}

	alpha, logProb, err_alpha := hmm.alpha(obs)
	if err_alpha != nil {
		t.Fatal(err_alpha)
	}

	beta, err_beta := hmm.beta(obs)
	if err_beta != nil {
		t.Fatal(err_beta)
	}

	t.Logf("LogProb: %f, Prob: %e\n", logProb, math.Exp(logProb))

	t.Logf("alpha:\n%+v\n", alpha)

	t.Logf("beta:\n%+v\n", beta)

}
