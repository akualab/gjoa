package hmm

import (
	"code.google.com/p/biogo.matrix"
	"github.com/akualab/gjoa/model"
	"github.com/akualab/gjoa/model/gaussian"
	//"math"
	"testing"
)

// Tests

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

// Test ColumnAt() function.
func TestColumnAt(t *testing.T) {

	mat, e := matrix.NewDense([][]float64{
		{10, 11, 12, 13}, {20, 21, 22, 23}, {30, 31, 32, 33}})
	if e != nil {
		t.Fatal(e)
	}

	col := ColumnAt(mat, 1)

	for i, expected := range []float64{11.0, 21.0, 31.0} {
		v := col.At(i, 0)
		if !model.Comparef64(expected, v) {
			t.Errorf("Wrong value. Expected: [%f], Got: [%f]", expected, v)
		}
	}
}

func MakeNewDenseMatrix(t *testing.T, m [][]float64) *matrix.Dense {
	mm, emm := matrix.NewDense(m)
	if emm != nil {
		t.Fatal(emm)
	}
	return mm
}

func MakeHMM(t *testing.T) *HMM {

	// Gaussian 1.
	mean1 := MakeNewDenseMatrix(t, [][]float64{{1}})
	var1 := MakeNewDenseMatrix(t, [][]float64{{1}})
	g1, eg1 := gaussian.NewGaussian(1, mean1, var1, true, true, "g1")
	if eg1 != nil {
		t.Fatal(eg1)
	}
	// Gaussian 2.
	mean2 := MakeNewDenseMatrix(t, [][]float64{{4}})
	var2 := MakeNewDenseMatrix(t, [][]float64{{4}})
	var2, ev2 := matrix.NewDense([][]float64{{4}})
	if ev2 != nil {
		t.Fatal(ev2)
	}
	g2, eg2 := gaussian.NewGaussian(1, mean2, var2, true, true, "g2")
	if eg2 != nil {
		t.Fatal(eg2)
	}

	initialStateProbs := MakeNewDenseMatrix(t, [][]float64{{0.8}, {0.2}})

	transProbs := MakeNewDenseMatrix(t, [][]float64{{0.9, 0.1}, {0.3, 0.7}})

	// These are the models.
	models := []*gaussian.Gaussian{g1, g2}

	// To pass the to an HMM we need to convert []*gaussian.Gaussian[]
	// to []model.Modeler
	// see http://golang.org/doc/faq#convert_slice_of_interface
	m := make([]model.Modeler, len(models))
	for i, v := range models {
		m[i] = v
	}
	hmm, e := NewHMM(transProbs, initialStateProbs, m)
	if e != nil {
		t.Fatal(e)
	}
	return hmm
}

var (
	obs0 = [][]float64{
		{0.1, 0.3, 1.1, 1.2, 0.7, 0.7, 5.5, 7.8, 10.0, 5.2, 1.1, 1.3}}
	alpha01 = []float64{
		-1.54708208451888,
		-2.80709238811418,
		-3.83134003758912,
		-4.86850442594034,
		-5.92973730403429,
		-6.99328952650412,
		-18.1370692144982,
		-36.3195887463382,
		-57.4758051059185,
		-32.2645657649804,
		-25.5978716740632,
		-26.5391830081456,
		-5.12277362619872,
		-6.99404330419337,
		-7.67194890763762,
		-8.58593275227677,
		-9.98735773434079,
		-11.0914094981902,
		-11.0792560557189,
		-14.8528937698143,
		-21.3216544274498,
		-23.4704150851531,
		-26.4904040834703,
		-29.0712307616184}
	beta01 = []float64{
		-24.9258011954291,
		-23.661415171904,
		-22.6397641116887,
		-21.6045197079498,
		-20.549461075003,
		-19.579339900188,
		-17.3294657178329,
		-13.5557050615525,
		-7.07931720879328,
		-2.06607429111337,
		-1.04620524392834,
		0,
		-25.9309053603105,
		-24.6182012641994,
		-23.5726285099546,
		-22.4540945910146,
		-20.5873818254665,
		-17.6335597285231,
		-15.3835555701327,
		-11.6097949124971,
		-5.14103425479379,
		-2.99265648819389,
		-1.76872378444132,
		0}
	gamma01 = []float64{
		-0.0101945977044363,
		-0.00581887777458709,
		-0.00841546703429051,
		-0.0103354516465726,
		-0.0165096967936958,
		-0.10994074444856,
		-9.00384625008746,
		-23.4126051256471,
		-38.0924336324682,
		-7.86795137385019,
		-0.181388235747997,
		-0.076494325902054,
		-4.59099030426571,
		-5.14955588614921,
		-4.78188873534866,
		-4.5773386610478,
		-4.11205087756371,
		-2.2622805444697,
		-0.000122943608043396,
		-6.79257761172257e-11,
		0,
		-0.000382891103450269,
		-1.79643918566812,
		-2.60854207937483}
	xsi = []float64{
		-0.0151076230417916, -0.0134668664218517, -0.0174701121578483, -0.0245758675622469,
		-0.11568757084122, -9.00936561095593, -29.3743846426695, -58.4605163217504,
		-42.9234897636508, -7.87738137552765, -0.204482040682161, 0,
		-5.32851547323339, -4.88295302258388, -4.71741675311881, -4.26911837592192,
		-2.37652915707246, -0.110077221151965, -9.0038462515104, -23.4126051256471,
		-38.1004437186275, -12.5365216739368, -3.96110379857833, 0,
		-4.68941145338974, -5.29903007116915, -4.95669127087446, -4.84061648256679,
		-5.27192028981584, -14.2060978713101, -23.4151837725584, -38.0924336338947,
		-7.86795137385019, -0.181842984368506, -2.1956267387574, 0,
		-6.95829686585791, -7.12399378960776, -6.612115474112, -6.04063655320304,
		-4.48823943832366, -2.26228704378271, -0.000122943675802531, -6.79259981618307e-11,
		-0.000382891103450158, -1.79646084505424, -2.90772605893015, 0}
)

func CompareSliceFloat(t *testing.T, expected []float64, actual []float64, message string) {
	for i, _ := range expected {
		if !model.Comparef64(expected[i], actual[i]) {
			t.Errorf("[%s]. Expected: [%f], Got: [%f]",
				message, expected[i], actual[i])
		}
	}
}

func CompareFloats(t *testing.T, expected float64, actual float64, message string) {
	if !model.Comparef64(expected, actual) {
		t.Errorf("[%s]. Expected: [%f], Got: [%f]",
			message, expected, actual)
	}
}

func TestEvaluationAlpha(t *testing.T) {

	hmm := MakeHMM(t)
	obs := MakeNewDenseMatrix(t, obs0)
	alpha, logProb, err_alpha := hmm.alpha(obs)
	if err_alpha != nil {
		t.Fatal(err_alpha)
	}
	expectedLogProb := -26.4626886822436
	CompareFloats(t, expectedLogProb, logProb, "Error in logProb")
	message := "Error in alpha"
	CompareSliceFloat(t, alpha01, alpha.ElementsVector(), message)
}

func TestEvaluationBeta(t *testing.T) {

	hmm := MakeHMM(t)
	obs := MakeNewDenseMatrix(t, obs0)
	beta, err_beta := hmm.beta(obs)
	if err_beta != nil {
		t.Fatal(err_beta)
	}
	message := "Error in beta"
	CompareSliceFloat(t, beta01, beta.ElementsVector(), message)
}

func TestEvaluationGamma(t *testing.T) {

	hmm := MakeHMM(t)
	obs := MakeNewDenseMatrix(t, obs0)
	alpha, _, err_alpha := hmm.alpha(obs)
	if err_alpha != nil {
		t.Fatal(err_alpha)
	}
	beta, err_beta := hmm.beta(obs)
	if err_beta != nil {
		t.Fatal(err_beta)
	}
	gamma, err_gamma := hmm.gamma(alpha, beta)
	if err_gamma != nil {
		t.Fatal(err_gamma)
	}
	message := "Error in gamma"
	CompareSliceFloat(t, gamma01, gamma.ElementsVector(), message)
}

func Convert3DSlideTo1D(s3 [][][]float64) []float64 {
	s1 := make([]float64, 0, 100)
	for _, v1 := range s3 {
		for _, v2 := range v1 {
			for _, v3 := range v2 {
				s1 = append(s1, v3)
			}
		}
	}
	return s1
}

func TestEvaluationXi(t *testing.T) {
	hmm := MakeHMM(t)
	obs := MakeNewDenseMatrix(t, obs0)
	alpha, _, err_alpha := hmm.alpha(obs)
	if err_alpha != nil {
		t.Fatal(err_alpha)
	}
	beta, err_beta := hmm.beta(obs)
	if err_beta != nil {
		t.Fatal(err_beta)
	}
	xi, err_xi := hmm.xi(obs, alpha, beta)
	if err_xi != nil {
		t.Fatal(err_xi)
	}
	xsi1 := Convert3DSlideTo1D(xi)
	message := "Error in xi"
	CompareSliceFloat(t, xsi, xsi1, message)
}
