// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

import (
	"testing"

	"github.com/akualab/gjoa/model"
)

type dummy struct{}

func (d dummy) LogProb(o model.Obs) float64 { return 0.0 }

func TestNet1(t *testing.T) {

	net := NewNetwork("test")
	s0 := net.AddEntryState()
	s1 := net.AddState(nil)
	s2 := net.AddState(nil)
	s3 := net.AddExitState()

	net.AddArc(s0, s1, 0.8)
	net.AddArc(s0, s2, 0.2)
	net.AddArc(s1, s1, 0.9)
	net.AddArc(s1, s2, 0.1)
	net.AddArc(s2, s2, 0.7)
	net.AddArc(s2, s3, 0.3)

	if !panics(func() { net.AddExitState() }) {
		t.Errorf("did not panic for duplicate exit state")
	}

	if !panics(func() { net.AddEntryState() }) {
		t.Errorf("did not panic for duplicate entry state")
	}

	net = NewNetwork("test")
	net.AddEntryState()
	net.AddState(nil)
	net.AddState(nil)

	err := net.Validate()
	if err == nil {
		t.Fatal("expected invalid network, missing exit state")
	}

	net = NewNetwork("test")
	net.AddExitState()
	net.AddState(nil)
	net.AddState(nil)

	err = net.Validate()
	if err == nil {
		t.Fatal("expected invalid network, missing entry state")
	}
	net.Init(10)
}

func TestNet2(t *testing.T) {

	net := NewNetwork("test")
	s0 := net.AddEntryState()
	s1 := net.AddState(dummy{})
	s2 := net.AddState(dummy{})
	s3 := net.AddExitState()

	net.AddArc(s0, s1, 0.8)
	net.AddArc(s0, s2, 0.2)
	net.AddArc(s1, s1, 0.9)
	net.AddArc(s1, s2, 0.1)
	net.AddArc(s2, s2, 0.7)
	net.AddArc(s2, s3, 0.3)
}

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

// func MakeNetwork(t *testing.T) *Model {

// 	// Gaussian 1.
// 	mean1 := []float64{1}
// 	sd1 := []float64{1}
// 	g1 := gm.NewModel(1, gm.Name("g1"), gm.Mean(mean1), gm.StdDev(sd1))

// 	// Gaussian 2.
// 	mean2 := []float64{4}
// 	sd2 := []float64{2}
// 	g2 := gm.NewModel(1, gm.Name("g2"), gm.Mean(mean2), gm.StdDev(sd2))

// 	//	initialStateProbs := []float64{0.8, 0.2}
// 	//	transProbs := [][]float64{{0.9, 0.1}, {0.3, 0.7}}

// 	net := NewNetwork("test")
// 	s0 := net.AddEntryState()
// 	s1 := net.AddState(g1)
// 	s2 := net.AddState(g2)
// 	s3 := net.AddExitState()

// 	net.AddArc(s0, s1, 0.8)
// 	net.AddArc(s0, s2, 0.2)
// 	net.AddArc(s1, s1, 0.9)
// 	net.AddArc(s1, s2, 0.1)
// 	net.AddArc(s2, s1, 0.3)
// 	net.AddArc(s2, s2, 0.7)

// 	_ = s3
// 	return NewModelFromNet(net)
// }

// func TestGraph(t *testing.T) {

// 	flag.Parse()
// 	hmm := MakeHMM(t)
// 	_, logProb := hmm.alpha(obs0)
// 	expectedLogProb := -26.4626886822436
// 	gjoa.CompareFloats(t, expectedLogProb, logProb, "Error in logProb", epsilon)
// }

func panics(fun func()) (b bool) {
	defer func() {
		err := recover()
		if err != nil {
			b = true
		}
	}()
	fun()
	return
}
