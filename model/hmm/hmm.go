/*

Hidden Markov model.


 α β γ ζ Φ

*/
package hmm

import (
	"bitbucket.org/akualab/gjoa/model"
	"code.google.com/p/biogo.matrix"
	"fmt"
	"github.com/golang/glog"
)

const ()

type HMM struct {

	// q(t) is the state at time t
	// states are labeled {0,2,...,N-1}

	// Number fo hidden states.
	// N
	nstates int

	// State-transition probability distribution matrix. [nstates x nstates]
	// a(i,j) = P[q(t+1) = j | q(t) = i]; 0 <= i,j <= N-1
	// when transitions between states are not allowed: a(i,j) = 0
	transProbs *matrix.Dense

	// Observation probability distribution functions. [nstates x 1]
	// b(j,k) = P[o(t) | q(t) = j]
	obsModels []model.Modeler

	// Initial state distribution. [nstates x 1]
	// π(i) = P[q(0) = i]; 0<=i<N
	initialStateProbs *matrix.Dense

	// Complete HMM model.
	// Φ = (A, B, π)
}

// Create a new HMM.
func NewHMM(transProbs, initialStateProbs *matrix.Dense, obsModels []model.Modeler) (hmm *HMM, e error) {

	r, c := transProbs.Dims()
	if r != c {
		e = fmt.Errorf("Matrix transProbs must be square. rows=[%d], cols=[%d]", r, c)
		return
	}

	hmm = &HMM{
		nstates:           r,
		transProbs:        transProbs,
		obsModels:         obsModels,
		initialStateProbs: initialStateProbs,
	}

	return
}

// Compute alphas. Indices are: α(state, time)
//
// α = | α(0,0),   α(0,1)   ... α(0,T-1)   |
//     | α(1,0),   α(1,1)   ... α(1,T-1)   |
//     ...
//     | α(N-1,0), α(N-1,1) ... α(N-1,T-1) |
//
//
// 1. Initialization: α(i,0) =  π(i) b(i,o(0)); 0<=i<N
// 2. Induction:      α(j,t+1) =  sum_{n=i}^{N-1}[α(i,t)a(i,j)] b(j,o(t+1)); 0<=t<T-1; 0<=j<N
// 3. Termination:    P(O/Φ) = sum_{i=0}^{N-1} α(i,T-1)
func (hmm *HMM) ComputeAlpha(observations *matrix.Dense) (e error) {

	// expected num rows: N = nstates
	// expected num cols: T
	N, T := observations.Dims()

	if N != hmm.nstates {
		e = fmt.Errorf("Mismatch between number of rows in observations matrix [%d] and number of states in the HMM [%d].", N, hmm.nstates)
		return
	}

	// Allocate log-alpha matrix.
	// TODO: use a reusable data structure to minimize garbage.
	α := matrix.MustDense(matrix.ZeroDense(N, T))

	// 1. Initialization.
	//α.SetColumn(0, hmm.initialStateProbs.Column(0))

	// 2. Induction.

	return
}
