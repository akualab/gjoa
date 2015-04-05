// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

import (
	"math/rand"
	"strconv"
	"testing"
	"time"

	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model"
	gm "github.com/akualab/gjoa/model/gaussian"
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

	return NewModel(OSet(ms), UpdateTP(true), UpdateOP(true))
}

func TestTrainBasic(t *testing.T) {

	data := [][]float64{{0.1}, {0.3}, {1.1}, {5.5}, {7.8}, {10.0}, {5.2}, {4.1}, {3.3}, {6.2}, {8.3}}

	m := makeHMM(t)
	h := m.Set.Nets[0]
	tp0 := narray.Exp(nil, h.A.Copy())
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

// should be equivalent to training a single gaussian, great for debugging.
func TestSingleState(t *testing.T) {

	// HMM to generate data.
	g01 := gm.NewModel(1, gm.Name("g01"), gm.Mean([]float64{0}), gm.StdDev([]float64{1}))

	h0 := narray.New(3, 3)
	h0.Set(1, 0, 1)
	h0.Set(.8, 1, 1)
	h0.Set(.2, 1, 2)
	h0 = narray.Log(nil, h0.Copy())

	ms0, _ := NewSet()
	net0, e0 := ms0.NewNet("hmm", h0,
		[]model.Modeler{nil, g01, nil})
	fatalIf(t, e0)
	hmm0 := NewModel(OSet(ms0))
	_ = hmm0

	// Create gaussian to estimate without using the HMM code.
	g := gm.NewModel(1, gm.Name("g1"), gm.Mean([]float64{-1}), gm.StdDev([]float64{2}))

	// Create initial HMM and estimate params from generated data.
	g1 := gm.NewModel(1, gm.Name("g1"), gm.Mean([]float64{-1}), gm.StdDev([]float64{2}))

	h := narray.New(3, 3)
	h.Set(1, 0, 1)
	h.Set(.5, 1, 1)
	h.Set(.5, 1, 2)
	h = narray.Log(nil, h.Copy())

	ms, _ = NewSet()
	net, e := ms.NewNet("hmm", h,
		[]model.Modeler{nil, g1, nil})
	fatalIf(t, e)
	hmm := NewModel(OSet(ms), UpdateTP(true), UpdateOP(true))

	iter := 5
	// number of sequences
	m := 1000
	numFrames := 0
	t0 := time.Now() // Start timer.
	for i := 0; i < iter; i++ {
		t.Logf("iter [%d]", i)

		// Make sure we generate the same data in each iteration.
		r := rand.New(rand.NewSource(33))
		gen := newGenerator(r, false, net0)

		// Reset all counters.
		hmm.Clear()
		g.Clear()

		// fix the seed to get the same sequence
		for j := 0; j < m; j++ {
			obs, states := gen.next("oid-" + fi(j))
			numFrames += len(states) - 2
			hmm.UpdateOne(obs, 1.0)

			// Update Gaussian
			for _, o := range obs.ValueAsSlice() {
				vec := o.([]float64)
				gobs := model.NewFloatObs(vec, model.SimpleLabel(""))
				g.UpdateOne(gobs, 1.0)
			}
		}
		hmm.Estimate()
		g.Estimate()
		t.Logf("iter:%d, hmm g1:   %+v", i, net.B[1])
		t.Logf("iter:%d, direct g1:%+v", i, g)
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
	t.Logf("hmm0 g1:%+v", net0.B[1])
	t.Logf("hmm  g1: %+v", net.B[1])

	t.Log("")
	t.Logf("direct g1:%+v", g)

	// Print time stats.
	t.Log("")
	t.Logf("Total time: %v", dur)
	t.Logf("Time per iteration: %v", dur/time.Duration(iter))
	t.Logf("Time per frame: %v", dur/time.Duration(iter*numFrames*m))

	gjoa.CompareSliceFloat(t, tp0.Data, tp.Data,
		"error in Trans Probs [0]", .03)

	CompareGaussians(t, net0.B[1].(*gm.Model), net.B[1].(*gm.Model), 0.03)

	if t.Failed() {
		t.FailNow()
	}

	// Recognize.
	sg := ms.SearchGraph()

	dec, e := graph.NewDecoder(sg)
	if e != nil {
		t.Fatal(e)
	}

	r := rand.New(rand.NewSource(5151))
	gen := newGenerator(r, true, net0)
	//	testDecoder(t, gen, dec, 1000)
	testDecoder(t, gen, dec, 10)
}

func TestHMMGauss(t *testing.T) {

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
	hmm0 := NewModel(OSet(ms0), UpdateTP(true), UpdateOP(true))
	_ = hmm0

	// Create random HMM and estimate params from obs.

	g1 := gm.NewModel(1, gm.Name("g1"), gm.Mean([]float64{-1}), gm.StdDev([]float64{2}))
	g2 := gm.NewModel(1, gm.Name("g2"), gm.Mean([]float64{18}), gm.StdDev([]float64{4}))

	h := narray.New(4, 4)
	h.Set(.5, 0, 1)
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
	hmm := NewModel(OSet(ms), UpdateTP(true), UpdateOP(true))

	iter := 10
	// number of sequences
	m := 500
	numFrames := 0
	t0 := time.Now() // Start timer.
	for i := 0; i < iter; i++ {
		t.Logf("iter [%d]", i)

		// Make sure we generate the same data in each iteration.
		r := rand.New(rand.NewSource(33))
		gen := newGenerator(r, false, net0)

		// Reset all counters.
		hmm.Clear()

		// fix the seed to get the same sequence
		for j := 0; j < m; j++ {
			obs, states := gen.next("oid-" + fi(j))
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
		"error in Trans Probs [0]", .03)

	CompareGaussians(t, net0.B[1].(*gm.Model), net.B[1].(*gm.Model), 0.03)
	CompareGaussians(t, net0.B[2].(*gm.Model), net.B[2].(*gm.Model), 0.03)

	if t.Failed() {
		t.FailNow()
	}

	// Recognize.
	g := ms.SearchGraph()

	dec, e := graph.NewDecoder(g)
	if e != nil {
		t.Fatal(e)
	}

	r := rand.New(rand.NewSource(5151))
	gen := newGenerator(r, true, net0)
	testDecoder(t, gen, dec, 1000)
}

func randomGaussian(r *rand.Rand, id string, dim int) *gm.Model {

	var mean, sd []float64
	startSD := 40.0
	for i := 0; i < dim; i++ {
		mean = append(mean, float64(r.Intn(10)*100.0))
		a := r.NormFloat64()*0.2 + 1.0 // pert 0.8 to 1.2
		sd = append(sd, startSD*a)
	}
	return gm.NewModel(dim, gm.Name(id), gm.Mean(mean), gm.StdDev(sd))
}

// creates a random gaussian by adding a perturbation to an existing gaussian.
func initGaussian(r *rand.Rand, m model.Modeler) *gm.Model {

	g := m.(*gm.Model)
	var mean, sd []float64
	for i := 0; i < g.ModelDim; i++ {
		a := r.NormFloat64()*0.2 + 1.0 // pert 0.8 to 1.2
		mean = append(mean, g.Mean[i]*a)
		sd = append(sd, g.StdDev[i]*a)
	}
	return gm.NewModel(g.ModelDim, gm.Name(g.ModelName), gm.Mean(mean), gm.StdDev(sd))
}

func addRandomNet(r *rand.Rand, ms *Set, id string, ns, dim int) (*Net, error) {

	m := []model.Modeler{nil}
	for i := 1; i < ns-1; i++ {
		sid := id + "-" + strconv.FormatInt(int64(i), 10)
		g := randomGaussian(r, sid, dim)
		m = append(m, g)
	}
	m = append(m, nil)
	h := MakeLeftToRight(ns, .5, 0)
	net, e := ms.NewNet(id, h, m)
	if e != nil {
		return nil, e
	}

	return net, nil
}

func initRandomSet(r *rand.Rand, in *Set) (*Set, error) {

	out, _ := NewSet()

	for q, h := range in.Nets {
		m := []model.Modeler{nil}
		for i := 1; i < h.ns-1; i++ {
			g := initGaussian(r, h.B[i])
			m = append(m, g)
		}
		m = append(m, nil)
		hn := randTrans(r, h.ns, false)
		id := "m" + strconv.FormatInt(int64(q), 10)
		_, e := out.NewNet(id, hn, m)
		if e != nil {
			return nil, e
		}
	}

	return out, nil
}

func TestHMMChain(t *testing.T) {

	r := rand.New(rand.NewSource(444))
	numModels := 20
	dim := 4 //8
	maxNumStates := 6
	iter0 := 1 // from alignments
	iter1 := 2 // FB
	ntrain0 := 1000
	ntrain1 := 20000
	maxChainLen := 8     // max number of nets in chain.
	numTestItems := 1000 // num test sequences.
	maxTestLen := 10

	// Create reference HMM to generate random sequences.
	ms0, _ := NewSet()
	for q := 0; q < numModels; q++ {
		ns := int(r.Intn(maxNumStates-2) + 3)
		id := "m" + strconv.FormatInt(int64(q), 10)
		net, e := addRandomNet(r, ms0, id, ns, dim)
		if e != nil {
			t.Fatal(e)
		}
		_ = net
	}
	hmm0 := NewModel(OSet(ms0), OAssign(DirectAssigner{}), UpdateTP(true), UpdateOP(true))
	t.Log("hmm0: ", hmm0)

	// Create random HMM and estimate params using the randomly generated sequences.
	ms, e := initRandomSet(r, ms0)
	if e != nil {
		t.Fatal(e)
	}
	//	hmm := NewModel(OSet(ms), OAssign(DirectAssigner{}), UpdateTP(true), UpdateOP(true))
	hmm := NewModel(OSet(ms), OAssign(DirectAssigner{}), UseAlignments(true))
	t.Log("initial hmm: ", hmm)

	numFrames := 0
	t0 := time.Now() // Start timer.
	t.Log("start training from alignments")
	for i := 0; i < iter0; i++ {
		t.Logf("iter [%d]", i)

		// Make sure we generate the same data in each iteration.
		r := rand.New(rand.NewSource(33))
		gen := newChainGen(r, true, maxChainLen, ms0.Nets...)

		// Reset all counters.
		hmm.Clear()

		// fix the seed to get the same sequence
		for j := 0; j < ntrain0; j++ {
			obs, states := gen.next("oid-" + fi(j))
			numFrames += len(states) - 2
			hmm.UpdateOne(obs, 1.0)
		}
		hmm.Estimate()
	}

	t.Log("start training using forward-backward algo")
	hmm.SetFlags(false, true, true)
	for i := 0; i < iter1; i++ {
		t.Logf("iter [%d]", i)

		// Make sure we generate the same data in each iteration.
		r := rand.New(rand.NewSource(55))
		gen := newChainGen(r, true, maxChainLen, ms0.Nets...)

		// Reset all counters.
		hmm.Clear()

		// fix the seed to get the same sequence
		for j := 0; j < ntrain1; j++ {
			obs, states := gen.next("oid-" + fi(j))
			numFrames += len(states) - 2
			hmm.UpdateOne(obs, 1.0)
		}
		hmm.Estimate()
	}

	dur := time.Now().Sub(t0)
	t.Log("dur: ", dur)

	for name, net0 := range ms0.byName {
		net := ms.byName[name]
		h0 := net0.A
		h := net.A
		tp0 := narray.Exp(nil, h0.Copy())
		tp := narray.Exp(nil, h.Copy())
		ns := tp.Shape[0]
		for i := 0; i < ns; i++ {
			for j := 0; j < ns; j++ {
				p0 := tp0.At(i, j)
				logp0 := h0.At(i, j)
				p := tp.At(i, j)
				logp := h.At(i, j)
				if p > smallNumber || p0 > smallNumber {
					t.Logf("name: %s, %d=>%d, p0:%5.2f, p:%5.2f, logp0:%8.5f, logp:%8.5f", name, i, j, p0, p, logp0, logp)
				}
			}
		}
		t.Log("")
		for i := 1; i < ns-1; i++ {
			t.Logf("hmm0 state:%d, %s", i, net0.B[i])
			t.Logf("hmm  state:%d, %s", i, net.B[i])
			t.Log("")
		}
	}

	t.Log("final hmm: ", hmm)

	// Print time stats.
	t.Log("")
	t.Logf("Total time: %v", dur)
	t.Logf("Time per iteration: %v", dur/time.Duration(iter1))
	t.Logf("Time per frame: %v", dur/time.Duration(iter1*numFrames*ntrain1))

	// Recognize.
	g := ms.SearchGraph()

	dec, e := graph.NewDecoder(g)
	if e != nil {
		t.Fatal(e)
	}

	r = rand.New(rand.NewSource(5151))
	gen := newChainGen(r, true, maxTestLen, ms0.Nets...)
	testDecoder(t, gen, dec, numTestItems)

}

func testDecoder(t *testing.T, gen sequencer, dec *graph.Decoder, numIterations int) {

	numErrors := 0
	n := 0
	for i := 0; i < numIterations; i++ {

		// Generate a sequence.
		obs, states := gen.next("oid-" + fi(i))
		t.Log("generated states: ", states)
		var refLabels []string
		for _, sid := range states {
			refLabels = append(refLabels, sid)
		}

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

		// for _, v := range best {
		// 	t.Logf("best: %+v", v)
		// }

		// Put the labels is a slice, exclude null nodes.
		hypLabels := best.Labels(true)

		t.Log("ref: ", refLabels)
		t.Log("hyp: ", hypLabels)

		// Compare labels
		if len(refLabels) != len(hypLabels) {
			t.Fatal("ref/hyp length mismath")
		}
		for k, lab := range refLabels {
			n++
			if lab != hypLabels[k] {
				numErrors++
			}
		}
	}
	relErr := float64(numErrors) / float64(n)
	t.Logf("num_labels:%d, num_errors:%d, rel_err:%5.1f%%", n, numErrors, relErr*100.0)
	maxErr := 0.03
	if relErr > maxErr {
		t.Fatalf("recognition error rate is %4.1f%% which is greater than %4.1f%%", relErr*100.0, maxErr*100.0)
	}
}

// func makeGmm(mean, sd [][]float64, weights []float64) *gmm.Model {

// 	dim := len(mean[0])
// 	ncomp := len(weights)

// 	g0 := gm.NewModel(ncomp, gm.Name("g0"), gm.Mean(mean[0]), gm.StdDev(sd[0]))
// 	g1 := gm.NewModel(ncomp, gm.Name("g1"), gm.Mean(mean[1]), gm.StdDev(sd[1]))
// 	components := []*gm.Model{g0, g1}
// 	return gmm.NewModel(dim, ncomp, gmm.Components(components), gmm.Weights(weights))
// }

// func TestTrainHmmGmm(t *testing.T) {
// 	var seed int64 = 31
// 	hmm0 := makeHmmGmm(t)
// 	hmm := makeRandomHmmGmm(t, seed)
// 	iter := 1
// 	// size of the generated sequence
// 	n := 100
// 	// number of sequences
// 	m := 1000
// 	//	eps := 0.03
// 	t0 := time.Now() // Start timer.
// 	for i := 0; i < iter; i++ {
// 		t.Logf("iter [%d]", i)

// 		// Reset all counters.
// 		hmm.Clear()

// 		// fix the seed to get the same sequence
// 		r := rand.New(rand.NewSource(33))
// 		gen := newGenerator(r, false, hmm0.Set.Nets[0])
// 		for j := 0; j < m; j++ {
// 			obs, states := gen.next("oid-" + fi(j))
// 			_ = states
// 			hmm.UpdateOne(obs, 1.0)
// 		}
// 		hmm.Estimate()
// 	}
// 	dur := time.Now().Sub(t0)
// 	//	CompareHMMs(t, hmm0, hmm, eps)
// 	h0 := hmm0.Set.Nets[0]
// 	h := hmm.Set.Nets[0]

// 	t.Logf("hmm0 - A:%+v, B[0]:%+v, B[1]:%+v", h0.A.Data, h0.B[0], h0.B[1])
// 	t.Logf("hmm  - A:%+v, B[0]:%+v, B[1]:%+v", h.A.Data, h.B[0], h.B[1])
// 	// Print time stats.
// 	t.Logf("Total time: %v", dur)
// 	t.Logf("Time per iteration: %v", dur/time.Duration(iter))
// 	t.Logf("Time per frame: %v", dur/time.Duration(iter*n*m))
// }

func CompareGaussians(t *testing.T, g1 *gm.Model, g2 *gm.Model, tol float64) {
	gjoa.CompareSliceFloat(t, g1.Mean, g2.Mean, "Wrong Mean", tol)
	gjoa.CompareSliceFloat(t, g1.StdDev, g2.StdDev, "Wrong SD", tol)
}

// func makeHmmGmm(t *testing.T) *Model {

// 	mean1 := [][]float64{{1, 2}, {5, 5}}
// 	sd1 := [][]float64{{0.5, 0.5}, {0.5, 0.5}}
// 	mean2 := [][]float64{{-1, -2}, {-6, -5}}
// 	sd2 := [][]float64{{0.5, 0.5}, {0.5, 0.5}}
// 	weight1 := []float64{0.6, 0.4}
// 	weight2 := []float64{0.3, 0.7}
// 	gmm1 := makeGmm(mean1, sd1, weight1)
// 	gmm2 := makeGmm(mean2, sd2, weight2)

// 	var err error
// 	h0 := narray.New(4, 4)
// 	h0.Set(.8, 0, 1)
// 	h0.Set(.2, 0, 2)
// 	h0.Set(.9, 1, 1)
// 	h0.Set(.1, 1, 2)
// 	h0.Set(.7, 2, 2)
// 	h0.Set(.3, 2, 3)
// 	h0 = narray.Log(nil, h0.Copy())

// 	ms, _ = NewSet()
// 	hmm0, err = ms.NewNet("hmm0", h0,
// 		[]model.Modeler{nil, gmm1, gmm2, nil})
// 	fatalIf(t, err)

// 	return NewModel(OSet(ms), UpdateTP(true), UpdateOP(true))
// }

// func makeRandomHmmGmm(t *testing.T, seed int64) *Model {
// 	mean := [][]float64{{2.5, 3}, {-2.5, -3}}
// 	sd := [][]float64{{0.7, 0.7}, {0.7, 0.7}}
// 	gmm1 := gmm.RandomModel(mean[0], sd[0], 2, "gmm1", seed)
// 	gmm2 := gmm.RandomModel(mean[1], sd[1], 2, "gmm2", seed)

// 	r := rand.New(rand.NewSource(seed))
// 	ran0 := r.Float64()
// 	ran1 := r.Float64()
// 	ran2 := r.Float64()

// 	var err error
// 	h0 := narray.New(4, 4)
// 	h0.Set(ran0, 0, 1)
// 	h0.Set(1-ran0, 0, 2)
// 	h0.Set(ran1, 1, 1)
// 	h0.Set(1-ran1, 1, 2)
// 	h0.Set(ran2, 2, 2)
// 	h0.Set(1-ran2, 2, 3)
// 	h0 = narray.Log(nil, h0.Copy())

// 	ms, _ = NewSet()
// 	_, err = ms.NewNet("random hmm", h0,
// 		[]model.Modeler{nil, gmm1, gmm2, nil})
// 	fatalIf(t, err)

// 	return NewModel(OSet(ms), UpdateTP(true), UpdateOP(true))
// }

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

func fi(i int) string {
	return strconv.FormatInt(int64(i), 10)
}
