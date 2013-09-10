package hmm

import (
	"github.com/akualab/gjoa/model"
	"github.com/akualab/gjoa/model/gaussian"
	"code.google.com/p/biogo.matrix"
	//"math"
	"testing"
)

// Tests

// Test ColumnAt() function.
func TestColumnAt(t *testing.T) {

	mat, e := matrix.NewDense([][]float64{
			{10, 11, 12, 13}, {20, 21, 22, 23}, {30, 31, 32, 33}})
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

func MakeHMM() (hmm *HMM, e error) {

	// Gaussian 1.
	mean1, _ := matrix.NewDense([][]float64{{1}})
	var1, _ := matrix.NewDense([][]float64{{1}})
	g1, _ := gaussian.NewGaussian(1, mean1, var1, true, true, "g1")

	// Gaussian 2.
	mean2, _ := matrix.NewDense([][]float64{{4}})
	var2, _ := matrix.NewDense([][]float64{{4}})
	g2, _ := gaussian.NewGaussian(1, mean2, var2, true, true, "g2")
	initialStateProbs, _ := matrix.NewDense([][]float64{{0.8}, {0.2}})
	transProbs, _ := matrix.NewDense([][]float64{{0.9, 0.1}, {0.3, 0.7}})

	// These are the models.
	models := []*gaussian.Gaussian{g1, g2}

	// To pass the to an HMM we need to convert []*gaussian.Gaussian[]
	// to []model.Modeler
	// see http://golang.org/doc/faq#convert_slice_of_interface
	m := make([]model.Modeler, len(models))
	for i, v := range models {
		m[i] = v
	}

	return NewHMM(transProbs, initialStateProbs, m)
}

var (
	obs, _ = matrix.NewDense([][]float64{
			{0.1, 0.3, 1.1, 1.2, 0.7, 0.7, 5.5, 7.8, 10.0, 5.2, 1.1, 1.3}})
	hmm, _ = MakeHMM()
)

func TestEvaluationAlpha(t *testing.T) {

	alpha, logProb, err_alpha := hmm.alpha(obs)
	if err_alpha != nil {
		t.Fatal(err_alpha)
	}
	t.Logf("logProb:\n%+v\n", logProb)
	expected := -26.53918
	actual := alpha.At(0,11)
    if !model.Comparef64(expected, actual) {
		t.Errorf("Wrong Alpha Expected: [%f], Got: [%f]", expected, actual )
	}
	expected = -29.07123076
	actual = alpha.At(1,11)
	if !model.Comparef64(expected, actual) {
        t.Errorf("Wrong Alpha Expected: [%f], Got: [%f]", expected, actual )
    }
}

func TestEvaluationBeta(t *testing.T) {

	beta, err_beta := hmm.beta(obs)
	if err_beta != nil {
		t.Fatal(err_beta)
	}
    /*
	gamma, err_gamma := hmm.gamma(alpha, beta)
	if err_gamma != nil {
		t.Fatal(err_gamma)
	}
    
	xi, err_xi := hmm.xi(obs, alpha, beta)
	if err_xi != nil {
		t.Fatal(err_xi)
	}
	*/


	t.Logf("beta:\n%+v\n", beta)

	//t.Logf("gamma:\n%+v\n", gamma)

	//t.Logf("xi:\n%+v\n", xi)

	/*
	   DISCUSSION:
	   If you look at the sample data and model params. I manufactured the
	   data as if it was emitted with the following sequence:

	   t:  0   1   2   3   4   5   6   7   8   9   10  11
	   q:  s0  s0  s0  s0  s0  s0  s1  s1  s1  s1  s0  s0
	   o:  0.1 0.3 1.1 1.2 0.7 0.7 5.5 7.8 10  5.2 1.1 1.3 <= 
	   data I created given the Gaussians [1,1] and [4,4]

	   I got the following gamma:

	   γ0: -0.03 -0.03 -0.05 -0.05 -0.04 -0.11 -9.02 -21 -36 -7.8 -0.15 -0.11
	   γ1: -3.35 -3.41 -3.01 -2.92 -3.13 -2.24 -0.00 -0  -0  -0   -1.91 -2.21

	   As you can see choosing the gamma with highest prob for each state give
	   us the hidden sequence of states.

	   gamma gives you the most likely state at time t. In this case the result is what we expect.

	   Viterbi gives you the P(q | O,  model), that is, it maximizes of over the whole sequence.
	*/
}
