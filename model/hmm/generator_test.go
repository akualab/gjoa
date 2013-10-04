package hmm

import (
	"github.com/akualab/gjoa/floatx"
	"github.com/akualab/gjoa/model"
	"github.com/akualab/gjoa/model/gaussian"
	"math/rand"
	"testing"
	"time"
)

func MakeHMM2(t *testing.T) *HMM {

	mean0 := []float64{1, 2}
	var0 := []float64{0.5, 0.3}
	mean1 := []float64{4, 4}
	var1 := []float64{0.2, 3}

	g0, eg0 := gaussian.NewGaussian(2, mean0, var0, true, true, "g0")
	if eg0 != nil {
		t.Fatal(eg0)
	}
	g1, eg1 := gaussian.NewGaussian(2, mean1, var1, true, true, "g1")
	if eg1 != nil {
		t.Fatal(eg1)
	}
	initialStateProbs := []float64{0.25, 0.75}
	transProbs := [][]float64{{0.7, 0.3}, {0.5, 0.5}}
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
	// This is for generating random means
	mm := []float64{1, 2}
	sd := []float64{0.5, 0.5}
	mean0, _ := model.RandNormalVector(mm, sd, r)
	mm = []float64{4, 4}
	mean1, _ := model.RandNormalVector(mm, sd, r)

	ran0 := r.Float64()
	ran1 := r.Float64()
	ran2 := r.Float64()
	ran3 := r.Float64()
	ran4 := r.Float64()
	var0 := []float64{ran3, 1 - ran3}
	var1 := []float64{ran4, 1 - ran4}
	g0, eg0 := gaussian.NewGaussian(2, mean0, var0, true, true, "g0")
	if eg0 != nil {
		t.Fatal(eg0)
	}
	g1, eg1 := gaussian.NewGaussian(2, mean1, var1, true, true, "g1")
	if eg1 != nil {
		t.Fatal(eg1)
	}
	initialStateProbs := []float64{ran0, 1 - ran0}
	transProbs := [][]float64{{ran1, 1 - ran1}, {ran2, 1 - ran2}}
	models := []*gaussian.Gaussian{g0, g1}
	m := make([]model.Trainer, len(models))
	for i, v := range models {
		m[i] = v
	}
	hmm, e := NewHMM(transProbs, initialStateProbs, m, true, "testhmm")
	if e != nil {
		t.Fatal(e)
	}
	return hmm
}

func TestTrainHMM(t *testing.T) {
	RunTestTrainHMM(t, true, true)
	RunTestTrainHMM(t, true, false)
	RunTestTrainHMM(t, false, true)
	RunTestTrainHMM(t, false, false)
}

func RunTestTrainHMM(t *testing.T, update_tp, update_ip bool) {

	hmm0 := MakeHMM2(t)
	hmm := MakeRandHMM(t, 35)
	hmm.trainingFlags.update_tp = update_tp
	hmm.trainingFlags.update_ip = update_ip
	log_ip := make([]float64, 0)
	copy(log_ip, hmm.logInitProbs)
	log_tp_0 := make([]float64, 0)
	log_tp_1 := make([]float64, 0)
	copy(log_tp_0, hmm.logTransProbs[0])
	copy(log_tp_1, hmm.logTransProbs[1])
	// number of updates
	iter := 5
	// size of the generated sequence
	n := 100
	// number of sequences
	m := 100000
	// max error for long test
	eps := 0.004
	if testing.Short() {
		m = 1000
		eps = 0.03
	}
	m00 := hmm0.obsModels[0].(*gaussian.Gaussian)
	m11 := hmm0.obsModels[1].(*gaussian.Gaussian)
	m0 := hmm.obsModels[0].(*gaussian.Gaussian)
	m1 := hmm.obsModels[1].(*gaussian.Gaussian)

	t0 := time.Now() // Start timer.
	for i := 0; i < iter; i++ {
		t.Logf("iter [%d]", i)
		// fix the seed to get the same sequence
		gen := MakeHMMGenerator(hmm0, 33)
		for j := 0; j < m; j++ {
			obs, _, err := gen.next(n)
			if err != nil {
				t.Fatal(err)
			}
			hmm.Update(obs, 1.0)
		}
		hmm.Estimate()
		// t.Logf here
		// Prepare for next iteration.
		hmm.Clear()
		// stats
		m0 = hmm.obsModels[0].(*gaussian.Gaussian)
		m1 = hmm.obsModels[1].(*gaussian.Gaussian)
		t.Logf("mean[0] %v, Variance %v", m0.Mean(), m0.Variance())
		t.Logf("mean[1] %v, Variance %v", m1.Mean(), m1.Variance())
		tmp := make([]float64, 2)
		floatx.Apply(exp, hmm.logTransProbs[0], tmp)
		t.Logf("transition prob [0] %v", tmp)
		floatx.Apply(exp, hmm.logTransProbs[1], tmp)
		t.Logf("transition prob [1] %v", tmp)
		floatx.Apply(exp, hmm.logInitProbs, tmp)
		t.Logf("logInitProbs %v", tmp)
	}
	dur := time.Now().Sub(t0)
	//var m0 *gaussian.Gaussian
	CompareGaussians(t, m00, m0, eps)
	CompareGaussians(t, m11, m1, eps)
	if update_tp {
		model.CompareSliceFloat(t, hmm0.logTransProbs[0], hmm.logTransProbs[0],
			"error in logTransProbs[0]", eps)
		model.CompareSliceFloat(t, hmm0.logTransProbs[1], hmm.logTransProbs[1],
			"error in logTransProbs[1]", eps)
	} else {
		model.CompareSliceFloat(t, log_tp_0, hmm.logTransProbs[0],
			"error in logTransProbs[0]", 0.0001)
		model.CompareSliceFloat(t, log_tp_1, hmm.logTransProbs[1],
			"error in logTransProbs[1]", 0.0001)
	}
	if update_ip {
		model.CompareSliceFloat(t, hmm0.logInitProbs, hmm.logInitProbs,
			"error in logInitProbs", eps)
	} else {
		model.CompareSliceFloat(t, log_ip, hmm.logInitProbs,
			"error in logInitProbs", 0.0001)
	}
	// Print time stats.
	t.Logf("Total time: %v", dur)
	t.Logf("Time per iteration: %v", dur/time.Duration(iter))
	t.Logf("Time per frame: %v", dur/time.Duration(iter*n*m))
}

func CompareGaussians(t *testing.T, g1 *gaussian.Gaussian, g2 *gaussian.Gaussian, eps float64) {
	model.CompareSliceFloat(t, g1.Mean(), g2.Mean(), "Wrong Mean", eps)
	model.CompareSliceFloat(t, g1.Variance(), g2.Variance(), "Wrong Variance", eps)
}
