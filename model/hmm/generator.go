// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

import (
	"math/rand"
	"strconv"

	"github.com/akualab/gjoa/model"
	"github.com/akualab/narray"
)

const seed = 33

// Generator generates random observations using an hmm model.
type generator struct {
	hmm    *Net
	r      *rand.Rand
	noNull bool
}

// NewGenerator returns an hmm data generator.
func newGenerator(r *rand.Rand, noNull bool, hmm *Net) *generator {
	return &generator{
		hmm:    hmm,
		r:      r,
		noNull: noNull,
	}
}

// Next returns the next observation sequence.
func (gen *generator) next() (model.FloatObsSequence, []string) {

	var data [][]float64
	name := gen.hmm.Name
	states := []string{name + "-0"}
	r := gen.r
	seq := model.NewFloatObsSequence(data, model.SimpleLabel(name), "").(model.FloatObsSequence)
	s := gen.hmm.nextState(0, r) // entry state
	states = append(states, name+"-"+strconv.FormatInt(int64(s), 10))
	for {
		g := gen.hmm.B[s]
		gs, ok := g.(model.Sampler)
		if !ok {
			panic("output PDF does not implement the sampler interface")
		}
		x := gs.Sample().(model.FloatObs)
		seq.Add(x, "")
		s = gen.hmm.nextState(s, r)
		states = append(states, name+"-"+strconv.FormatInt(int64(s), 10))
		if s == gen.hmm.ns-1 {
			// Reached exit state.
			break
		}
	}
	if gen.noNull {
		states = states[1 : len(states)-1]
	}
	return seq, states
}

// Generator generates random observations using a chain of hmm model.
type chainGen struct {
	hmms   []*Net
	r      *rand.Rand
	q      int
	noNull bool
}

func newChainGen(r *rand.Rand, noNull bool, nets ...*Net) *chainGen {

	return &chainGen{
		hmms:   nets,
		r:      r,
		noNull: noNull,
	}
}

// next returns the next chain of observation sequences.
// The output is organized as a slice of obs and a slice of state
// sequences. Each element of the slice corresponds to a model
// in the input chain.
func (gen *chainGen) next() (*model.FloatObsSequence, []string) {

	var obs []*model.FloatObsSequence
	var states []string

	for _, h := range gen.hmms {
		g := newGenerator(gen.r, gen.noNull, h)
		o, s := g.next()
		obs = append(obs, &o)
		for _, one := range s {
			states = append(states, one)
		}
	}

	fos := model.JoinFloatObsSequence(obs...).(*model.FloatObsSequence)
	return fos, states
}

// randTrans generates a left-to right random transition prob matrix.
// n is the total number of states including entry/exit.
func randTrans(r *rand.Rand, n int) *narray.NArray {

	if n < 3 {
		panic("need at least 3 states")
	}

	a := narray.New(n, n)

	// state 0
	if n > 3 {
		p := getProbs(r, 2)
		a.Set(p[0], 0, 1) // entry
		a.Set(p[1], 0, 2) // skip first emmiting state
	} else {
		a.Set(1, 0, 1) // entry
	}

	// states 1..n-3
	for i := 1; i < n-2; i++ {
		p := getProbs(r, 3)
		a.Set(p[0], i, i)   // self loop
		a.Set(p[1], i, i+1) // to right
		a.Set(p[2], i, i+2) // skip
	}

	// state ns-2
	p := getProbs(r, 2)
	a.Set(p[0], n-2, n-2) // self
	a.Set(p[1], n-2, n-1) // to exit (no skip)

	return narray.Log(nil, a)
}

// Get n random probabilities. Adds to one.
func getProbs(r *rand.Rand, n int) []float64 {

	na := narray.Rand(r, n)
	d := 1.0 / na.Sum()
	narray.Scale(na, na, d)
	return na.Data
}
