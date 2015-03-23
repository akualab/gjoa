// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

import (
	"math/rand"
	"strings"
	"testing"
	"time"

	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model"
	gm "github.com/akualab/gjoa/model/gaussian"
	"github.com/akualab/gjoa/model/gmm"
	"github.com/akualab/graph"
	"github.com/akualab/narray"
)

func makeHMM(t *testing.T) *Model {

	// Gaussian 1.
	mean1 := []float64{1}
	sd1 := []float64{1}
	g1 := gm.NewModel(1, gm.Name("g1"), gm.Mean(mean1), gm.StdDev(sd1))

	// Gaussian 2.
	mean2 := []float64{4}
	sd2 := []float64{2}
	g2 := gm.NewModel(1, gm.Name("g2"), gm.Mean(mean2), gm.StdDev(sd2))

	//	initialStateProbs := []float64{0.8, 0.2}
	//	transProbs := [][]float64{{0.9, 0.1}, {0.3, 0.7}}

	var err error
	h0 := narray.New(4, 4)
	//	h0.Set(.8, 0, 1)
	h0.Set(1, 0, 1)
	//	h0.Set(.2, 0, 2)
	h0.Set(.5, 1, 1)
	h0.Set(.5, 1, 2)
	h0.Set(.7, 2, 2)
	h0.Set(.3, 2, 3)
	h0 = narray.Log(nil, h0.Copy())

	ms, _ = NewSet()
	_, err = ms.NewNet("hmm0", h0,
		[]model.Modeler{nil, g1, g2, nil})
	fatalIf(t, err)

	return NewModel(OSet(ms))
}

func TestTrainBasic(t *testing.T) {

	data := [][]float64{{0.1}, {0.3}, {1.1}, {5.5}, {7.8}, {10.0}, {5.2}, {4.1}, {3.3}, {6.2}, {8.3}}

	m := makeHMM(t)
	h := m.Set.Nets[0]
	tp0 := narray.Exp(nil, h.A.Copy())
	//	obs := model.NewFloatObsSequence(obs0, model.SimpleLabel(""), "")
	obs := model.NewFloatObsSequence(data, model.SimpleLabel(""), "")

	m.Clear()
	m.UpdateOne(obs, 1.0)
	m.Estimate()

	m.Clear()
	m.UpdateOne(obs, 1.0)
	m.Estimate()

	tp := narray.Exp(nil, h.A.Copy())
	ns := tp.Shape[0]
	for i := 0; i < ns; i++ {
		for j := 0; j < ns; j++ {
			p0 := tp0.At(i, j)
			p := tp.At(i, j)
			if p > smallNumber || p0 > smallNumber {
				t.Logf("TP: %d=>%d, p0:%5.2f, p:%5.2f", i, j, p0, p)
			}
		}
	}

	t.Log("")
	t.Logf("hmm  g1: %+v, g2:%+v", h.B[1], h.B[2])
}

func makeGmm(mean, sd [][]float64, weights []float64) *gmm.Model {

	dim := len(mean[0])
	ncomp := len(weights)

	g0 := gm.NewModel(ncomp, gm.Name("g0"), gm.Mean(mean[0]), gm.StdDev(sd[0]))
	g1 := gm.NewModel(ncomp, gm.Name("g1"), gm.Mean(mean[1]), gm.StdDev(sd[1]))
	components := []*gm.Model{g0, g1}
	return gmm.NewModel(dim, ncomp, gmm.Components(components), gmm.Weights(weights))
}

func makeHmmGmm(t *testing.T) *Model {

	mean1 := [][]float64{{1, 2}, {5, 5}}
	sd1 := [][]float64{{0.5, 0.5}, {0.5, 0.5}}
	mean2 := [][]float64{{-1, -2}, {-6, -5}}
	sd2 := [][]float64{{0.5, 0.5}, {0.5, 0.5}}
	weight1 := []float64{0.6, 0.4}
	weight2 := []float64{0.3, 0.7}
	gmm1 := makeGmm(mean1, sd1, weight1)
	gmm2 := makeGmm(mean2, sd2, weight2)

	var err error
	h0 := narray.New(4, 4)
	h0.Set(.8, 0, 1)
	h0.Set(.2, 0, 2)
	h0.Set(.9, 1, 1)
	h0.Set(.1, 1, 2)
	h0.Set(.7, 2, 2)
	h0.Set(.3, 2, 3)
	h0 = narray.Log(nil, h0.Copy())

	ms, _ = NewSet()
	hmm0, err = ms.NewNet("hmm0", h0,
		[]model.Modeler{nil, gmm1, gmm2, nil})
	fatalIf(t, err)

	return NewModel(OSet(ms))
}

func makeRandomHmmGmm(t *testing.T, seed int64) *Model {
	mean := [][]float64{{2.5, 3}, {-2.5, -3}}
	sd := [][]float64{{0.7, 0.7}, {0.7, 0.7}}
	gmm1 := gmm.RandomModel(mean[0], sd[0], 2, "gmm1", seed)
	gmm2 := gmm.RandomModel(mean[1], sd[1], 2, "gmm2", seed)

	r := rand.New(rand.NewSource(seed))
	ran0 := r.Float64()
	ran1 := r.Float64()
	ran2 := r.Float64()

	var err error
	h0 := narray.New(4, 4)
	h0.Set(ran0, 0, 1)
	h0.Set(1-ran0, 0, 2)
	h0.Set(ran1, 1, 1)
	h0.Set(1-ran1, 1, 2)
	h0.Set(ran2, 2, 2)
	h0.Set(1-ran2, 2, 3)
	h0 = narray.Log(nil, h0.Copy())

	ms, _ = NewSet()
	_, err = ms.NewNet("random hmm", h0,
		[]model.Modeler{nil, gmm1, gmm2, nil})
	fatalIf(t, err)

	return NewModel(OSet(ms))
}

func TestTrainHmmGaussian(t *testing.T) {

	// Create reference HMM to generate observations.

	g01 := gm.NewModel(1, gm.Name("g01"), gm.Mean([]float64{0}), gm.StdDev([]float64{1}))
	g02 := gm.NewModel(1, gm.Name("g02"), gm.Mean([]float64{16}), gm.StdDev([]float64{2}))

	h0 := narray.New(4, 4)
	h0.Set(.6, 0, 1)
	h0.Set(.4, 0, 2)
	h0.Set(.9, 1, 1)
	h0.Set(.1, 1, 2)
	h0.Set(.7, 2, 2)
	h0.Set(.3, 2, 3)
	h0 = narray.Log(nil, h0.Copy())

	ms0, _ := NewSet()
	net0, e0 := ms0.NewNet("hmm", h0,
		[]model.Modeler{nil, g01, g02, nil})
	fatalIf(t, e0)
	hmm0 := NewModel(OSet(ms0))
	_ = hmm0

	// Create random HMM and estimate params from obs.

	g1 := gm.NewModel(1, gm.Name("g1"), gm.Mean([]float64{-1}), gm.StdDev([]float64{2}))
	g2 := gm.NewModel(1, gm.Name("g2"), gm.Mean([]float64{18}), gm.StdDev([]float64{4}))

	h := narray.New(4, 4)
	h.Set(1, 0, 1)
	h.Set(.5, 0, 2)
	h.Set(.5, 1, 1)
	h.Set(.5, 1, 2)
	h.Set(.5, 2, 2)
	h.Set(.5, 2, 3)
	h = narray.Log(nil, h.Copy())

	ms, _ = NewSet()
	net, e := ms.NewNet("hmm", h,
		[]model.Modeler{nil, g1, g2, nil})
	fatalIf(t, e)
	hmm := NewModel(OSet(ms))

	iter := 20
	// number of sequences
	m := 500
	numFrames := 0
	gen := newGenerator(net0)
	t0 := time.Now() // Start timer.
	for i := 0; i < iter; i++ {
		t.Logf("iter [%d]", i)

		// Reset all counters.
		hmm.Clear()

		// fix the seed to get the same sequence
		for j := 0; j < m; j++ {
			obs, states := gen.next()
			numFrames += len(states) - 2
			hmm.UpdateOne(obs, 1.0)
		}
		hmm.Estimate()
	}
	dur := time.Now().Sub(t0)
	tp0 := narray.Exp(nil, h0.Copy())
	tp := narray.Exp(nil, net.A.Copy())
	ns := tp.Shape[0]
	for i := 0; i < ns; i++ {
		for j := 0; j < ns; j++ {
			p0 := tp0.At(i, j)
			logp0 := h0.At(i, j)
			p := tp.At(i, j)
			logp := h.At(i, j)
			if p > smallNumber || p0 > smallNumber {
				t.Logf("TP: %d=>%d, p0:%5.2f, p:%5.2f, logp0:%8.5f, logp:%8.5f", i, j, p0, p, logp0, logp)
			}
		}
	}

	t.Log("")
	t.Logf("hmm0 g1:%+v, g2:%+v", net0.B[1], net0.B[2])
	t.Logf("hmm  g1: %+v, g2:%+v", net.B[1], net.B[2])

	// Print time stats.
	t.Log("")
	t.Logf("Total time: %v", dur)
	t.Logf("Time per iteration: %v", dur/time.Duration(iter))
	t.Logf("Time per frame: %v", dur/time.Duration(iter*numFrames*m))

	gjoa.CompareSliceFloat(t, tp0.Data, tp.Data,
		"error in Trans Probs [0]", .05)

	CompareGaussians(t, net0.B[1].(*gm.Model), net.B[1].(*gm.Model), 0.05)
	CompareGaussians(t, net0.B[2].(*gm.Model), net.B[2].(*gm.Model), 0.05)

	if t.Failed() {
		t.FailNow()
	}

	// Recognize.
	g := ms.SearchGraph()

	dec, e := graph.NewDecoder(g)
	if e != nil {
		t.Fatal(e)
	}

	for i := 0; i < 1000; i++ {
		// Generate a sequence.
		obs, states := gen.next()
		t.Log("generated states: ", states)
		refLabels := strings.Split(string(obs.Label().(model.SimpleLabel)), ",")

		// Find the optimnal sequence.
		token := dec.Decode(obs.ValueAsSlice())

		// The token has the backtrace to find the optimal path.
		if token == nil {
			t.Fatalf(">>>> got nil token for ref: %s", refLabels)
		} else {
			t.Logf(">>>> backtrace: %s", token.PrintBacktrace())
		}

		// Get the best hyp.
		best := token.Best()

		for _, v := range best {
			t.Logf("best: %+v", v)
		}

		// Put the labels is a slice, exclude null nodes.
		hypLabels := best.Labels(true)

		t.Log("ref: ", refLabels)
		t.Log("hyp: ", hypLabels)

		// Compare labels
		if len(refLabels) != len(hypLabels) {
			t.Errorf("ref/hyp length mismath")
		} else {
			for k, lab := range refLabels {
				if lab != hypLabels[k] {
					t.Error("label mismath")
					continue
				}
			}
		}
		if t.Failed() {
			t.FailNow()
		}
	}
}

func TestTrainHmmGmm(t *testing.T) {
	var seed int64 = 31
	hmm0 := makeHmmGmm(t)
	hmm := makeRandomHmmGmm(t, seed)
	iter := 1
	// size of the generated sequence
	n := 100
	// number of sequences
	m := 1000
	//	eps := 0.03
	t0 := time.Now() // Start timer.
	for i := 0; i < iter; i++ {
		t.Logf("iter [%d]", i)

		// Reset all counters.
		hmm.Clear()

		// fix the seed to get the same sequence
		gen := newGenerator(hmm0.Set.Nets[0])
		for j := 0; j < m; j++ {
			obs, states := gen.next()
			_ = states
			hmm.UpdateOne(obs, 1.0)
		}
		hmm.Estimate()
	}
	dur := time.Now().Sub(t0)
	//	CompareHMMs(t, hmm0, hmm, eps)
	h0 := hmm0.Set.Nets[0]
	h := hmm.Set.Nets[0]

	t.Logf("hmm0 - A:%+v, B[0]:%+v, B[1]:%+v", h0.A.Data, h0.B[0], h0.B[1])
	t.Logf("hmm  - A:%+v, B[0]:%+v, B[1]:%+v", h.A.Data, h.B[0], h.B[1])
	// Print time stats.
	t.Logf("Total time: %v", dur)
	t.Logf("Time per iteration: %v", dur/time.Duration(iter))
	t.Logf("Time per frame: %v", dur/time.Duration(iter*n*m))
}

func CompareGaussians(t *testing.T, g1 *gm.Model, g2 *gm.Model, eps float64) {
	gjoa.CompareSliceFloat(t, g1.Mean, g2.Mean, "Wrong Mean", eps)
	gjoa.CompareSliceFloat(t, g1.StdDev, g2.StdDev, "Wrong SD", eps)
}

// func CompareHMMs(t *testing.T, hmm0 *Model, hmm *Model, eps float64) {

// 	gjoa.CompareSliceFloat(t, hmm0.Set.Nets[0], hmm.TransProbs[0],
// 		"error in TransProbs[0]", eps)
// 	gjoa.CompareSliceFloat(t, hmm0.TransProbs[1], hmm.TransProbs[1],
// 		"error in TransProbs[1]", eps)
// 	gjoa.CompareSliceFloat(t, hmm0.InitProbs, hmm.InitProbs,
// 		"error in logInitProbs", eps)
// 	mA := hmm0.ObsModels[0].(*gmm.Model)
// 	mB := hmm0.ObsModels[1].(*gmm.Model)
// 	m0 := hmm.ObsModels[0].(*gmm.Model)
// 	m1 := hmm.ObsModels[1].(*gmm.Model)
// 	if DistanceGmm2(t, mA, m0) < DistanceGmm2(t, mA, m1) {
// 		CompareGMMs(t, mA, m0, eps)
// 		CompareGMMs(t, mB, m1, eps)
// 	} else {
// 		CompareGMMs(t, mA, m1, eps)
// 		CompareGMMs(t, mB, m0, eps)
// 	}
// }

// func CompareGMMs(t *testing.T, g1 *gmm.Model, g2 *gmm.Model, eps float64) {
// 	d0 := DistanceGaussian(t, g1.Components[0], g2.Components[0])
// 	d1 := DistanceGaussian(t, g1.Components[0], g2.Components[1])
// 	t.Logf("distance between gaussians: %f %f", d0, d1)
// 	if d0 < d1 {
// 		CompareGaussians(t, g1.Components[0], g2.Components[0], eps)
// 		CompareGaussians(t, g1.Components[1], g2.Components[1], eps)
// 		gjoa.CompareSliceFloat(t, g1.Weights, g2.Weights, "Wrong Weights", eps)
// 	} else {
// 		CompareGaussians(t, g1.Components[0], g2.Components[1], eps)
// 		CompareGaussians(t, g1.Components[1], g2.Components[0], eps)
// 		w := []float64{g1.Weights[1], g1.Weights[0]}
// 		gjoa.CompareSliceFloat(t, w, g2.Weights, "Wrong Weights", eps)
// 	}
// }

// // distance for GMM with two components
// func DistanceGmm2(t *testing.T, g1 *gmm.Model, g2 *gmm.Model) float64 {
// 	distance0 := DistanceGaussian(t, g1.Components[0], g2.Components[0])
// 	distance0 += DistanceGaussian(t, g1.Components[1], g2.Components[1])
// 	distance1 := DistanceGaussian(t, g1.Components[0], g2.Components[1])
// 	distance1 += DistanceGaussian(t, g1.Components[1], g2.Components[0])
// 	return math.Min(distance0, distance1)
// }

// // L_inf distance between gaussian means
// func DistanceGaussian(t *testing.T, g1 *gm.Model, g2 *gm.Model) float64 {
// 	arr0 := g1.Mean
// 	arr1 := g2.Mean
// 	err := 0.0
// 	for i, _ := range arr0 {
// 		err = math.Max(err, math.Abs(arr0[i]-arr1[i]))
// 	}
// 	return err
// }
