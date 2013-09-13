package hmm

import (
	"flag"
	"github.com/akualab/gjoa/floatx"
	"github.com/akualab/gjoa/model"
	"github.com/akualab/gjoa/model/gaussian"
	"testing"
)

func init() {
	flag.Set("logtostderr", "true")
	flag.Set("v", "4")
}

// Tests

/*
   DISCUSSION:
   We created a simple 2-state HMM for testing.

   If you look at the sample data and model params. I manufactured the
   data as if it was emitted with the following sequence:

   t:  0   1   2   3   4   5   6   7   8   9   10  11
   q:  s0  s0  s0  s0  s0  s0  s1  s1  s1  s1  s0  s0
   o:  0.1 0.3 1.1 1.2 0.7 0.7 5.5 7.8 10  5.2 1.1 1.3 <=
   data I created given the Gaussians [1,1] and [4,4]

   I got the following gamma:

   γ0: -0.01 -0.00 -0.01 -0.01 -0.02 -0.11 -9.00 -23 -38 -7.8 -0.18 -0.08
   γ1: -4.59 -5.15 -4.78 -4.58 -4.11 -2.26 -0.00 -0  -0  -0   -1.80 -2.60

   As you can see choosing the gamma with highest prob for each state give
   us the hidden sequence of states.

   gamma gives you the most likely state at time t. In this case the result is what we expect.

   Viterbi gives you the P(q | O,  model), that is, it maximizes of over the whole sequence.
*/

func MakeHMM(t *testing.T) *HMM {

	// Gaussian 1.
	mean1 := []float64{1}
	var1 := []float64{1}
	g1, eg1 := gaussian.NewGaussian(1, mean1, var1, true, true, "g1")
	if eg1 != nil {
		t.Fatal(eg1)
	}
	// Gaussian 2.
	mean2 := []float64{4}
	var2 := []float64{4}
	g2, eg2 := gaussian.NewGaussian(1, mean2, var2, true, true, "g2")
	if eg2 != nil {
		t.Fatal(eg2)
	}

	initialStateProbs := []float64{0.8, 0.2}
	transProbs := [][]float64{{0.9, 0.1}, {0.3, 0.7}}

	// These are the models.
	models := []*gaussian.Gaussian{g1, g2}

	// To pass the to an HMM we need to convert []*gaussian.Gaussian[]
	// to []model.Modeler
	// see http://golang.org/doc/faq#convert_slice_of_interface
	m := make([]model.Modeler, len(models))
	for i, v := range models {
		m[i] = v
	}
	hmm, e := NewHMM(transProbs, initialStateProbs, m, false, "testhmm")
	if e != nil {
		t.Fatal(e)
	}
	return hmm
}

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

func CompareSliceInt(t *testing.T, expected []int, actual []int, message string) {
	for i, _ := range expected {
		if expected[i] != actual[i] {
			t.Errorf("[%s]. Expected: [%d], Got: [%d]",
				message, expected[i], actual[i])
		}
	}
}

func TestLogProb(t *testing.T) {

	flag.Parse()
	hmm := MakeHMM(t)
	_, logProb, err_alpha := hmm.alpha(obs0)
	if err_alpha != nil {
		t.Fatal(err_alpha)
	}
	expectedLogProb := -26.4626886822436
	CompareFloats(t, expectedLogProb, logProb, "Error in logProb")
}

func TestEvaluationGamma(t *testing.T) {

	flag.Parse()
	hmm := MakeHMM(t)
	alpha, _, err_alpha := hmm.alpha(obs0)
	if err_alpha != nil {
		t.Fatal(err_alpha)
	}
	beta, err_beta := hmm.beta(obs0)
	if err_beta != nil {
		t.Fatal(err_beta)
	}
	gamma, err_gamma := hmm.gamma(alpha, beta)
	if err_gamma != nil {
		t.Fatal(err_gamma)
	}
	message := "Error in gamma"
	CompareSliceFloat(t, gamma01, floatx.Flatten2D(gamma), message)
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

	flag.Parse()
	hmm := MakeHMM(t)
	alpha, _, err_alpha := hmm.alpha(obs0)
	if err_alpha != nil {
		t.Fatal(err_alpha)
	}
	beta, err_beta := hmm.beta(obs0)
	if err_beta != nil {
		t.Fatal(err_beta)
	}
	xi, err_xi := hmm.xi(obs0, alpha, beta)
	if err_xi != nil {
		t.Fatal(err_xi)
	}
	xsi1 := Convert3DSlideTo1D(xi)
	message := "Error in xi"
	CompareSliceFloat(t, xsi, xsi1, message)
}

func TestViterbi(t *testing.T) {
	flag.Parse()
	hmm := MakeHMM(t)
	bt, logProbViterbi, err := hmm.viterbi(obs0)
	if err != nil {
		t.Fatal(err)
	}
	expectedViterbiLog := -26.8129904950932
	CompareFloats(t, expectedViterbiLog, logProbViterbi, "Error in logProbViterbi")
	CompareSliceInt(t, viterbiSeq, bt, "Error in viterbi seq")
}

var (
	obs0 = [][]float64{
		{0.1, 0.3, 1.1, 1.2, 0.7, 0.7, 5.5, 7.8, 10.0, 5.2, 1.1, 1.3}}
	viterbiSeq = []int{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0}
	gamma01    = []float64{
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
