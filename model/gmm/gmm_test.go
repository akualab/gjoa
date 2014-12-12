// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmm

import (
	"flag"
	"math/rand"
	"os"
	"testing"

	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model"
	"github.com/akualab/gjoa/model/gaussian"
)

const epsilon = 0.004

func init() {
	flag.Set("logtostderr", "true")
	flag.Set("v", "2")
}

func TestGMMName(t *testing.T) {

	gmm := NewModel(4, 123, Name("mygmm"))

	// for i, c := range gmm.Components() {
	// 	t.Logf("Name for comp #%4d: %s", i, c.Name())
	// }
	name := gmm.Components[111].Name()
	if name != "mygmm-111" {
		t.Errorf("Wrong component name in gmm. Expected: [mygmm-111], Got: [%s]", name)
	}
}

// Trains a GMM as follows:
// 1 - Estimate a Gaussian model params for the training set.
// 2 - Use the mean and sd of the training set to generate
//     a random GMM to be used as seed.
// 3 - Run several iterations of the GMM max likelihood training algorithm
//     to estimate the GMM weights and the Gaussian component mean, and
//     variance vectors.
func TestTrainGMM(t *testing.T) {

	var seed int64 = 33
	numComp := 2
	numIter := 10
	numObs := 1000000

	mean0 := []float64{1, 2}
	std0 := []float64{0.3, 0.3}
	mean1 := []float64{4, 4}
	std1 := []float64{1, 1}
	dim := len(mean0)
	gmm := NewModel(dim, numComp, Name("mygmm"))
	t.Logf("Initial Weights: \n%+v", gmm.Weights)
	{
		// Estimate mean variance of the data.
		g := gaussian.NewModel(dim, gaussian.Name("test training"))

		r := rand.New(rand.NewSource(seed))
		for i := 0; i < numObs; i++ {
			rv, err := model.RandNormalVector(mean0, std0, r)
			if err != nil {
				t.Fatal(err)
			}
			g.UpdateOne(model.F64ToObs(rv), 1.0)
			rv, err = model.RandNormalVector(mean1, std1, r)
			if err != nil {
				t.Fatal(err)
			}
			g.UpdateOne(model.F64ToObs(rv), 1.0)
		}
		g.Estimate()
		t.Logf("Gaussian Model for training set:")
		t.Logf("Mean: \n%+v", g.Mean)
		t.Logf("SD: \n%+v", g.StdDev)

		// Use the estimated mean and sd to generate a seed GMM.
		gmm = RandomModel(g.Mean, g.StdDev, numComp,
			"mygmm", 99)
		t.Logf("Random GMM: %+v.", gmm)
		t.Logf("Component 0: %+v.", gmm.Components[0])
		t.Logf("Component 1: %+v.", gmm.Components[1])
	}

	for iter := 0; iter < numIter; iter++ {
		t.Logf("Starting GMM training iteration %d.", iter)

		// Reset the same random number generator to make sure we use the
		// same observations in each iterations.
		r := rand.New(rand.NewSource(seed))

		// Update GMM stats.
		for i := 0; i < numObs; i++ {
			rv, err := model.RandNormalVector(mean0, std0, r)
			if err != nil {
				t.Fatal(err)
			}
			gmm.UpdateOne(model.F64ToObs(rv), 1.0)
			rv, err = model.RandNormalVector(mean1, std1, r)
			if err != nil {
				t.Fatal(err)
			}
			gmm.UpdateOne(model.F64ToObs(rv), 1.0)
		}

		// Estimates GMM params.
		gmm.Estimate()

		t.Logf("Iter: %d", iter)
		t.Logf("GMM: %+v", gmm)
		t.Logf("Weights: \n%+v", gmm.Weights)
		t.Logf("Likelihood: %f", gmm.Likelihood)
		t.Logf("Num Samples: %f", gmm.NSamples)
		for _, c := range gmm.Components {
			t.Logf("%s: Mean: \n%+v", c.Name(), c.Mean)
			t.Logf("%s: STD: \n%+v", c.Name(), c.StdDev)
		}

		// Prepare for next iteration.
		gmm.Clear()
	}

	for i := 0; i < dim; i++ {
		g := gmm.Components[1]
		if !gjoa.Comparef64(mean0[i], g.Mean[i], epsilon) {
			t.Errorf("Wrong Mean[%d]. Expected: [%f], Got: [%f]",
				i, mean0[i], g.Mean[i])
		}
		if !gjoa.Comparef64(std0[i], g.StdDev[i], epsilon) {
			t.Errorf("Wrong STD[%d]. Expected: [%f], Got: [%f]",
				i, std0[i], g.StdDev[i])
		}
	}

	for i := 0; i < dim; i++ {
		g := gmm.Components[0]
		if !gjoa.Comparef64(mean1[i], g.Mean[i], epsilon) {
			t.Errorf("Wrong Mean[%d]. Expected: [%f], Got: [%f]",
				i, mean1[i], g.Mean[i])
		}
		if !gjoa.Comparef64(std1[i], g.StdDev[i], epsilon) {
			t.Errorf("Wrong STD[%d]. Expected: [%f], Got: [%f]",
				i, std1[i], g.StdDev[i])
		}
	}

	if !gjoa.Comparef64(0.5, gmm.Weights[0], epsilon) {
		t.Errorf("Wrong weights[0]. Expected: [%f], Got: [%f]",
			0.5, gmm.Weights[0])
	}

	if !gjoa.Comparef64(0.5, gmm.Weights[1], epsilon) {
		t.Errorf("Wrong weights[0]. Expected: [%f], Got: [%f]",
			0.5, gmm.Weights[1])
	}

}

func MakeGMM(t *testing.T) *Model {

	mean0 := []float64{1, 2}
	sd0 := []float64{0.3, 0.3}
	mean1 := []float64{4, 4}
	sd1 := []float64{1, 1}
	weights := []float64{0.6, 0.4}
	dim := len(mean0)

	g0 := gaussian.NewModel(2, gaussian.Name("g0"), gaussian.Mean(mean0), gaussian.StdDev(sd0))
	g1 := gaussian.NewModel(2, gaussian.Name("g1"), gaussian.Mean(mean1), gaussian.StdDev(sd1))
	components := []*gaussian.Model{g0, g1}
	gmm := NewModel(dim, 2, Name("mygmm"), Components(components), Weights(weights))
	return gmm
}

// Another version of previous test.
func TestTrainGMM2(t *testing.T) {
	dim := 2
	numComp := 2
	numIter := 10
	numObs := 2000000
	gmm0 := MakeGMM(t)
	gmm := NewModel(dim, numComp, Name("mygmm"))

	t.Logf("Initial Weights: \n%+v", gmm.Weights)
	mean01 := []float64{2.5, 3}
	sd01 := []float64{0.70710678118, 0.70710678118}
	gmm = RandomModel(mean01, sd01, numComp, "mygmm", 99)

	for iter := 0; iter < numIter; iter++ {
		t.Logf("Starting GMM training iteration %d.", iter)

		// Reset all counters..
		gmm.Clear()

		for i := 0; i < numObs; i++ {
			// random from gmm0
			//			rv := gmm0.Sample().(model.FloatObs)
			//			gmm.UpdateOne(rv.Value().([]float64), 1.0)
			rv := gmm0.Sample()
			gmm.UpdateOne(rv, 1.0)

		}
		gmm.Estimate()

		t.Logf("Iter: %d", iter)
		t.Logf("GMM: %+v", gmm)
		t.Logf("Weights: \n%+v", gmm.Weights)
		t.Logf("Likelihood: %f", gmm.Likelihood)
		t.Logf("Num Samples: %f", gmm.NSamples)
		for _, c := range gmm.Components {
			t.Logf("%s: Mean: \n%+v", c.Name(), c.Mean)
			t.Logf("%s: STD: \n%+v", c.Name(), c.StdDev)
		}

	}
	// Checking results
	// The components can be in different orders
	if gjoa.Comparef64(1.0, gmm.Components[0].Mean[0], epsilon) {
		CompareGaussians(t, gmm0.Components[0], gmm.Components[0], epsilon)
		CompareGaussians(t, gmm0.Components[1], gmm.Components[1], epsilon)
	} else {
		CompareGaussians(t, gmm0.Components[1], gmm.Components[0], epsilon)
		CompareGaussians(t, gmm0.Components[0], gmm.Components[1], epsilon)
	}
}

func TestWriteReadGMM(t *testing.T) {
	dim := 2
	numComp := 2
	numIter := 10
	numObs := 10000
	gmm0 := MakeGMM(t)
	gmm := NewModel(dim, numComp)
	t.Logf("Initial Weights: \n%+v", gmm.Weights)
	mean01 := []float64{2.5, 3}
	sd01 := []float64{0.70710678118, 0.70710678118}
	gmm = RandomModel(mean01, sd01, numComp, "mygmm", 99)
	for iter := 0; iter < numIter; iter++ {
		t.Logf("Starting GMM training iteration %d.", iter)
		gmm.Clear()

		for i := 0; i < numObs; i++ {
			rv := gmm0.Sample()
			gmm.UpdateOne(rv, 1.0)
		}
		gmm.Estimate()

		t.Logf("Iter: %d", iter)
	}

	// Write model.
	fn := os.TempDir() + "gmm.json"
	t.Logf("Wrote to temp file: %s\n", fn)
	gmm.WriteFile(fn)
	/*
		x, e1 := gmm.ReadFile(fn)
		if e1 != nil {
			t.Fatal(e1)
		}
		gmm1 := x.(*GMM)
	*/

	// Create another Gaussian model.
	gmm1, e1 := ReadFile(fn)
	if e1 != nil {
		t.Fatal(e1)
	}

	// Compare gmm and gmm1
	for k, v := range gmm.Components {
		CompareGaussians(t, v, gmm1.Components[k], epsilon)
	}
	gjoa.CompareSliceFloat(t, gmm.Weights, gmm1.Weights, "Weights don't match.", epsilon)
	gjoa.CompareSliceFloat(t, gmm.PosteriorSum, gmm1.PosteriorSum, "PosteriorSum doesn't match.", epsilon)
}

func CompareGaussians(t *testing.T, g1 *gaussian.Model, g2 *gaussian.Model, epsilon float64) {
	gjoa.CompareSliceFloat(t, g1.Mean, g2.Mean, "Wrong Mean", epsilon)
	gjoa.CompareSliceFloat(t, g1.StdDev, g2.StdDev, "Wrong SD", epsilon)
}
