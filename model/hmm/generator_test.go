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
	"github.com/akualab/gjoa/floatx"
	"github.com/akualab/gjoa/model"
	"github.com/akualab/gjoa/model/gaussian"
)

func MakeHMM2(t *testing.T) *Model {

	mean0 := []float64{1, 2}
	sd0 := []float64{0.70710678118, 0.5477225575}
	mean1 := []float64{4, 4}
	sd1 := []float64{0.4472135955, 1.73205080757}

	g0 := gaussian.NewModel(2, gaussian.Name("g0"), gaussian.Mean(mean0), gaussian.StdDev(sd0))
	g1 := gaussian.NewModel(2, gaussian.Name("g1"), gaussian.Mean(mean1), gaussian.StdDev(sd1))

	initialStateProbs := []float64{0.25, 0.75}
	transProbs := [][]float64{{0.7, 0.3}, {0.5, 0.5}}
	models := []*gaussian.Model{g0, g1}
	m := make([]model.Modeler, len(models))
	for i, v := range models {
		m[i] = v
	}

	return NewModel(transProbs, m, InitProbs(initialStateProbs), Seed(0))
}

func MakeRandHMM(t *testing.T, seed int64) *Model {
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
	sd0 := []float64{math.Sqrt(ran3), math.Sqrt(1 - ran3)}
	sd1 := []float64{math.Sqrt(ran4), math.Sqrt(1 - ran4)}
	g0 := gaussian.NewModel(2, gaussian.Name("g0"), gaussian.Mean(mean0), gaussian.StdDev(sd0))
	g1 := gaussian.NewModel(2, gaussian.Name("g1"), gaussian.Mean(mean1), gaussian.StdDev(sd1))

	initialStateProbs := []float64{ran0, 1 - ran0}
	transProbs := [][]float64{{ran1, 1 - ran1}, {ran2, 1 - ran2}}
	models := []*gaussian.Model{g0, g1}
	m := make([]model.Modeler, len(models))
	for i, v := range models {
		m[i] = v
	}

	return NewModel(transProbs, m, InitProbs(initialStateProbs), Seed(0))
}

func TestTrainHMM(t *testing.T) {

	hmm0 := MakeHMM2(t)
	hmm := MakeRandHMM(t, 35)
	log_ip := make([]float64, 0)
	copy(log_ip, hmm.InitProbs)
	log_tp_0 := make([]float64, 0)
	log_tp_1 := make([]float64, 0)
	copy(log_tp_0, hmm.TransProbs[0])
	copy(log_tp_1, hmm.TransProbs[1])
	// number of updates
	iter := 5
	// size of the generated sequence
	n := 100
	// number of sequences
	m := 100000
	// max error for long test
	eps := 0.007
	if testing.Short() {
		m = 1000
		eps = 0.03
	}
	m00 := hmm0.ObsModels[0].(*gaussian.Model)
	m11 := hmm0.ObsModels[1].(*gaussian.Model)
	m0 := hmm.ObsModels[0].(*gaussian.Model)
	m1 := hmm.ObsModels[1].(*gaussian.Model)

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

		// stats
		m0 = hmm.ObsModels[0].(*gaussian.Model)
		m1 = hmm.ObsModels[1].(*gaussian.Model)
		t.Logf("mean[0] %v, SD %v", m0.Mean, m0.StdDev)
		t.Logf("mean[1] %v, SD %v", m1.Mean, m1.StdDev)
		tmp := make([]float64, 2)
		floatx.Exp(tmp, hmm.TransProbs[0])
		t.Logf("transition prob [0] %v", tmp)
		floatx.Exp(tmp, hmm.TransProbs[1])
		t.Logf("transition prob [1] %v", tmp)
		floatx.Exp(tmp, hmm.InitProbs)
		t.Logf("logInitProbs %v", tmp)
	}
	dur := time.Now().Sub(t0)
	CompareGaussians(t, m00, m0, eps)
	CompareGaussians(t, m11, m1, eps)
	gjoa.CompareSliceFloat(t, hmm0.TransProbs[0], hmm.TransProbs[0],
		"error in TransProbs[0]", eps)
	gjoa.CompareSliceFloat(t, hmm0.TransProbs[1], hmm.TransProbs[1],
		"error in TransProbs[1]", eps)

	gjoa.CompareSliceFloat(t, hmm0.InitProbs, hmm.InitProbs,
		"error in logInitProbs", eps)
	// Print time stats.
	t.Logf("Total time: %v", dur)
	t.Logf("Time per iteration: %v", dur/time.Duration(iter))
	t.Logf("Time per frame: %v", dur/time.Duration(iter*n*m))

}

func CompareGaussians(t *testing.T, g1 *gaussian.Model, g2 *gaussian.Model, eps float64) {
	gjoa.CompareSliceFloat(t, g1.Mean, g2.Mean, "Wrong Mean", eps)
	gjoa.CompareSliceFloat(t, g1.StdDev, g2.StdDev, "Wrong SD", eps)
}
