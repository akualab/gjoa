// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model"
	gm "github.com/akualab/gjoa/model/gaussian"
	"github.com/akualab/gjoa/model/gmm"
)

func MakeGmm(t *testing.T, mean, sd [][]float64, weights []float64) *gmm.Model {

	dim := len(mean[0])
	ncomp := len(weights)

	g0 := gm.NewModel(ncomp, gm.Name("g0"), gm.Mean(mean[0]), gm.StdDev(sd[0]))
	g1 := gm.NewModel(ncomp, gm.Name("g1"), gm.Mean(mean[1]), gm.StdDev(sd[1]))
	components := []*gm.Model{g0, g1}
	return gmm.NewModel(dim, ncomp, gmm.Components(components), gmm.Weights(weights))
}

func MakeHmmGmm(t *testing.T) *Model {

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
	models := []*gmm.Model{gmm0, gmm1}
	m := make([]model.Modeler, len(models))
	for i, v := range models {
		m[i] = v
	}
	return NewModel(transProbs, m, InitProbs(initialStateProbs))
}

func MakeRandomHmmGmm(t *testing.T, seed int64) *Model {
	mean := [][]float64{{2.5, 3}, {-2.5, -3}}
	sd := [][]float64{{0.7, 0.7}, {0.7, 0.7}}

	gmm0 := gmm.RandomModel(mean[0], sd[0], 2, "gmm0", seed)
	gmm1 := gmm.RandomModel(mean[1], sd[1], 2, "gmm1", seed)

	r := rand.New(rand.NewSource(seed))
	ran0 := r.Float64()
	ran1 := r.Float64()
	ran2 := r.Float64()
	initialStateProbs := []float64{ran0, 1 - ran0}
	transProbs := [][]float64{{ran1, 1 - ran1}, {ran2, 1 - ran2}}
	models := []*gmm.Model{gmm0, gmm1}
	m := make([]model.Modeler, len(models))
	for i, v := range models {
		m[i] = v
	}
	return NewModel(transProbs, m, InitProbs(initialStateProbs))
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
		gen := NewGenerator(hmm0)
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

func CompareHMMs(t *testing.T, hmm0 *Model, hmm *Model, eps float64) {
	gjoa.CompareSliceFloat(t, hmm0.TransProbs[0], hmm.TransProbs[0],
		"error in TransProbs[0]", eps)
	gjoa.CompareSliceFloat(t, hmm0.TransProbs[1], hmm.TransProbs[1],
		"error in TransProbs[1]", eps)
	gjoa.CompareSliceFloat(t, hmm0.InitProbs, hmm.InitProbs,
		"error in logInitProbs", eps)
	mA := hmm0.ObsModels[0].(*gmm.Model)
	mB := hmm0.ObsModels[1].(*gmm.Model)
	m0 := hmm.ObsModels[0].(*gmm.Model)
	m1 := hmm.ObsModels[1].(*gmm.Model)
	if DistanceGmm2(t, mA, m0) < DistanceGmm2(t, mA, m1) {
		CompareGMMs(t, mA, m0, eps)
		CompareGMMs(t, mB, m1, eps)
	} else {
		CompareGMMs(t, mA, m1, eps)
		CompareGMMs(t, mB, m0, eps)
	}
}

func CompareGMMs(t *testing.T, g1 *gmm.Model, g2 *gmm.Model, eps float64) {
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
func DistanceGmm2(t *testing.T, g1 *gmm.Model, g2 *gmm.Model) float64 {
	distance0 := DistanceGaussian(t, g1.Components[0], g2.Components[0])
	distance0 += DistanceGaussian(t, g1.Components[1], g2.Components[1])
	distance1 := DistanceGaussian(t, g1.Components[0], g2.Components[1])
	distance1 += DistanceGaussian(t, g1.Components[1], g2.Components[0])
	return math.Min(distance0, distance1)
}

// L_inf distance between gaussian means
func DistanceGaussian(t *testing.T, g1 *gm.Model, g2 *gm.Model) float64 {
	arr0 := g1.Mean
	arr1 := g2.Mean
	err := 0.0
	for i, _ := range arr0 {
		err = math.Max(err, math.Abs(arr0[i]-arr1[i]))
	}
	return err
}
