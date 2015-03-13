// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

import (
	"math/rand"
	"strconv"

	"github.com/akualab/gjoa/model"
)

const seed = 33

// Generator generates random observations using an hmm model.
type generator struct {
	hmm *Net
	r   *rand.Rand
}

// NewGenerator returns an hmm data generator.
func newGenerator(hmm *Net) *generator {
	r := rand.New(rand.NewSource(seed))
	return &generator{
		hmm: hmm,
		r:   r,
	}
}

// Next returns the next observation sequence.
func (gen *generator) next() (model.FloatObsSequence, []int) {

	var data [][]float64
	states := []int{0}
	r := gen.r
	seq := model.NewFloatObsSequence(data, "", "").(model.FloatObsSequence)
	s := gen.hmm.nextState(0, r) // entry state
	states = append(states, s)
	for {
		g := gen.hmm.B[s]
		gs, ok := g.(model.Sampler)
		if !ok {
			panic("output PDF does not implement the sampler interface")
		}

		x := gs.Sample().(model.FloatObs)
		lab := gen.hmm.Name + "-" + strconv.FormatInt(int64(s), 10)
		seq.Add(x, lab)
		s = gen.hmm.nextState(s, r)
		states = append(states, s)
		if s == gen.hmm.ns-1 {
			// Reached exit state.
			break
		}
	}
	return seq, states
}
