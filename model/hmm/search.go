// Copyright (c) 2015 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

import (
	"math"
	"strconv"

	"github.com/akualab/gjoa/model"
	"github.com/akualab/graph"
)

// Value is the value type associated with graph nodes. Must implement the
// graph.Viterbier interface.
type nodeValue struct {
	scorer      model.Modeler
	entry, exit bool
	net         *Net
}

// MakeGraph creates a graph based on a set of HMM networks.
func MakeGraph(set *Set) *graph.Graph {

	return nil
}

// Graph converts an HMM Net object to a graph where states are nodes. Like
// an HMM net, the graph contains a single entry and exit non-emmiting state.
func (m *Net) Graph() *graph.Graph {

	g := graph.New()
	for i := 0; i < m.ns; i++ {
		fromKey := m.Name + strconv.FormatInt(int64(i), 10)
		nv := nodeValue{scorer: m.B[i], net: m}
		g.Set(fromKey, nv)
		for j := i; j < m.ns; j++ {
			if m.A.At(i, j) > math.Inf(-1) {
				toKey := m.Name + strconv.FormatInt(int64(j), 10)
				g.Connect(fromKey, toKey, m.A.At(i, j))
			}
		}
	}
	return g
}

// The viterbi algorithm computes the probable sequence of states for an HMM.
// These are the equations in log scale:
//
// delta(j, t) = max_{z_1,... z_{t-1}} log P(z_1,..,z_{t-1}, z_t=j | x_1,...,x_t )
//
// Recursion in log scale   delta(j, t) [nstates x T]
// delta(j, 0) = π(j) + b(j, 0)    for j in [0, N-1]
// delta(j, t) = max_k [ delta(k, t-1) + a(k, j) + b(j, t) ]   j in [0, N-1], t in [1, T-1]
// index(j, t) = argmax_k [ delta(k, t-1) + a(k,j) + b(j, t) ] j in [0, N-1], t in [1, T-1]
//
// Decoding z* is the output sequence [Tx1]
// z*(T-1) = argmax_j delta(j, T-1)
// z*(t) = index(z*(t+1), t+1)  t in [0, T-2]
// logProb = max_j delta(j, T-1)
//
// func (hmm *Model) Viterbi(observations [][]float64) (bt []int, logViterbiProb float64, e error) {

// 	// Num states
// 	N := hmm.NStates

// 	// num rows: T
// 	// num cols: numElements
// 	T, _ := floatx.Check2D(observations)

// 	// Allocate delta, index and bt
// 	delta := floatx.MakeFloat2D(N, T)
// 	index := make([][]int, N)
// 	bt = make([]int, T)
// 	for i := 0; i < N; i++ {
// 		index[i] = make([]int, T)
// 	}

// 	// Init delta
// 	for i := 0; i < N; i++ {
// 		o := model.F64ToObs(observations[0])
// 		b := hmm.ObsModels[i].LogProb(o)
// 		delta[i][0] = hmm.InitProbs[i] + b
// 	}

// 	// Recursion
// 	for t := 1; t < T; t++ {
// 		for i := 0; i < N; i++ {
// 			// Computing max in k to define delta(i,t)
// 			// init max with k=0
// 			o := model.F64ToObs(observations[t])
// 			b := hmm.ObsModels[i].LogProb(o)
// 			max := delta[0][t-1] + hmm.TransProbs[0][i] + b
// 			argmax := 0
// 			for k := 1; k < N; k++ {
// 				tempProb := delta[k][t-1] + hmm.TransProbs[k][i] + b
// 				if tempProb > max {
// 					max = tempProb
// 					argmax = k
// 				}
// 			}
// 			delta[i][t] = max
// 			index[i][t] = argmax
// 		}
// 	}

// 	// Decoding
// 	// init
// 	max := delta[0][T-1]
// 	argmax := 0
// 	for i := 1; i < N; i++ {
// 		if delta[i][T-1] > max {
// 			max = delta[i][T-1]
// 			argmax = i
// 		}
// 	}
// 	bt[T-1] = argmax
// 	logViterbiProb = max

// 	for t := T - 2; t >= 0; t-- {
// 		bt[t] = index[bt[t+1]][t+1]
// 	}

// 	return
// }
