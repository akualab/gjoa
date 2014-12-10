package hmm

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model"
	gm "github.com/akualab/gjoa/model/gaussian"
)

func MakeGmm(t *testing.T, mean, sd [][]float64, weights []float64) *gm.GMM {

	dim := len(mean[0])
	ncomp := len(weights)

	g0 := gm.NewGaussian(gm.GaussianParam{
		NumElements: ncomp,
		Name:        "g0",
		Mean:        mean[0],
		StdDev:      sd[0],
	})

	g1 := gm.NewGaussian(gm.GaussianParam{
		NumElements: ncomp,
		Name:        "g1",
		Mean:        mean[1],
		StdDev:      sd[1],
	})

	components := []*gm.Gaussian{g0, g1}
	gmm := gm.NewGMM(gm.GMMParam{
		NumElements:   dim,
		NumComponents: ncomp,
		Name:          "mygmm",
	})

	// this should probably be done in NewGaussianMixture
	gmm.Components = components
	gmm.Weights = weights
	return gmm
}

func MakeHmmGmm(t *testing.T) *HMM {

	mean0 := [][]float64{{1, 2}, {5, 5}}
	sd0 := [][]float64{{0.5, 0.5}, {0.5, 0.5}}
	mean1 := [][]float64{{-1, -2}, {-6, -5}}
	sd1 := [][]float64{{0.5, 0.5}, {0.5, 0.5}}
	weight0 := []float64{0.6, 0.4}
	weight1 := []float64{0.3, 0.7}
	gmm0 := MakeGmm(t, mean0, sd0, weight0)
	gmm1 := MakeGmm(t, mean1, sd1, weight1)
	initialStateProbs := []float64{0.3, 0.7}
	transProbs := [][]float64{{0.6, 0.4}, {0.5, 0.5}}
	models := []*gm.GMM{gmm0, gmm1}
	m := make([]model.Modeler, len(models))
	for i, v := range models {
		m[i] = v
	}
	hmm, e := NewHMM(HMMParam{
		TransProbs:        transProbs,
		InitialStateProbs: initialStateProbs,
		ObsModels:         m,
		Name:              "testhmm",
		UpdateIP:          true,
		UpdateTP:          true,
		GeneratorSeed:     0,
		GeneratorMaxLen:   100,
	})

	if e != nil {
		t.Fatal(e)
	}
	return hmm
}

func MakeRandomHmmGmm(t *testing.T, seed int64) *HMM {
	mean := [][]float64{{2.5, 3}, {-2.5, -3}}
	sd := [][]float64{{0.7, 0.7}, {0.7, 0.7}}

	gmm0 := gm.RandomGMM(mean[0], sd[0], 2, "gmm0", seed)
	gmm1 := gm.RandomGMM(mean[1], sd[1], 2, "gmm1", seed)

	r := rand.New(rand.NewSource(seed))
	ran0 := r.Float64()
	ran1 := r.Float64()
	ran2 := r.Float64()
	initialStateProbs := []float64{ran0, 1 - ran0}
	transProbs := [][]float64{{ran1, 1 - ran1}, {ran2, 1 - ran2}}
	models := []*gm.GMM{gmm0, gmm1}
	m := make([]model.Modeler, len(models))
	for i, v := range models {
		m[i] = v
	}
	hmm, e := NewHMM(HMMParam{
		TransProbs:        transProbs,
		InitialStateProbs: initialStateProbs,
		ObsModels:         m,
		Name:              "testhmm0",
		UpdateIP:          true,
		UpdateTP:          true,
		GeneratorSeed:     0,
		GeneratorMaxLen:   100,
	})
	if e != nil {
		t.Fatal(e)
	}
	return hmm
}

func TestTrainHmmGmm(t *testing.T) {
	var seed int64 = 31
	hmm0 := MakeHmmGmm(t)
	hmm := MakeRandomHmmGmm(t, seed)
	iter := 6
	// size of the generated sequence
	n := 100
	// number of sequences
	m := 1000
	eps := 0.03
	t0 := time.Now() // Start timer.
	for i := 0; i < iter; i++ {
		t.Logf("iter [%d]", i)

		// Reset all counters.
		hmm.Clear()

		// fix the seed to get the same sequence
		gen := NewGenerator(hmm0, 33)
		for j := 0; j < m; j++ {
			obs, _, err := gen.Next(n)
			if err != nil {
				t.Fatal(err)
			}
			hmm.UpdateOne(obs, 1.0)
		}
		hmm.Estimate()
	}
	dur := time.Now().Sub(t0)
	CompareHMMs(t, hmm0, hmm, eps)
	// Print time stats.
	t.Logf("Total time: %v", dur)
	t.Logf("Time per iteration: %v", dur/time.Duration(iter))
	t.Logf("Time per frame: %v", dur/time.Duration(iter*n*m))
}

func CompareHMMs(t *testing.T, hmm0 *HMM, hmm *HMM, eps float64) {
	gjoa.CompareSliceFloat(t, hmm0.TransProbs[0], hmm.TransProbs[0],
		"error in TransProbs[0]", eps)
	gjoa.CompareSliceFloat(t, hmm0.TransProbs[1], hmm.TransProbs[1],
		"error in TransProbs[1]", eps)
	gjoa.CompareSliceFloat(t, hmm0.InitProbs, hmm.InitProbs,
		"error in logInitProbs", eps)
	mA := hmm0.ObsModels[0].(*gm.GMM)
	mB := hmm0.ObsModels[1].(*gm.GMM)
	m0 := hmm.ObsModels[0].(*gm.GMM)
	m1 := hmm.ObsModels[1].(*gm.GMM)
	if DistanceGmm2(t, mA, m0) < DistanceGmm2(t, mA, m1) {
		CompareGMMs(t, mA, m0, eps)
		CompareGMMs(t, mB, m1, eps)
	} else {
		CompareGMMs(t, mA, m1, eps)
		CompareGMMs(t, mB, m0, eps)
	}
}

func CompareGMMs(t *testing.T, g1 *gm.GMM, g2 *gm.GMM, eps float64) {
	d0 := DistanceGaussian(t, g1.Components[0], g2.Components[0])
	d1 := DistanceGaussian(t, g1.Components[0], g2.Components[1])
	t.Logf("distance between gaussians: %f %f", d0, d1)
	if d0 < d1 {
		CompareGaussians(t, g1.Components[0], g2.Components[0], eps)
		CompareGaussians(t, g1.Components[1], g2.Components[1], eps)
		gjoa.CompareSliceFloat(t, g1.Weights, g2.Weights, "Wrong Weights", eps)
	} else {
		CompareGaussians(t, g1.Components[0], g2.Components[1], eps)
		CompareGaussians(t, g1.Components[1], g2.Components[0], eps)
		w := []float64{g1.Weights[1], g1.Weights[0]}
		gjoa.CompareSliceFloat(t, w, g2.Weights, "Wrong Weights", eps)
	}
}

// distance for GMM with two components
func DistanceGmm2(t *testing.T, g1 *gm.GMM, g2 *gm.GMM) float64 {
	distance0 := DistanceGaussian(t, g1.Components[0], g2.Components[0])
	distance0 += DistanceGaussian(t, g1.Components[1], g2.Components[1])
	distance1 := DistanceGaussian(t, g1.Components[0], g2.Components[1])
	distance1 += DistanceGaussian(t, g1.Components[1], g2.Components[0])
	return math.Min(distance0, distance1)
}

// L_inf distance between gaussian means
func DistanceGaussian(t *testing.T, g1 *gm.Gaussian, g2 *gm.Gaussian) float64 {
	arr0 := g1.Mean
	arr1 := g2.Mean
	err := 0.0
	for i, _ := range arr0 {
		err = math.Max(err, math.Abs(arr0[i]-arr1[i]))
	}
	return err
}
