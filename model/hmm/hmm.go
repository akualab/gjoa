/*

Hidden Markov model.


 α β γ ζ Φ

*/
package hmm

import (
	"bitbucket.org/akualab/gjoa/model"
	"code.google.com/p/biogo.matrix"
	"fmt"
	"math"
	//"github.com/golang/glog"
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
	// => saved in log domain.
	transProbs *matrix.Dense

	// Observation probability distribution functions. [nstates x 1]
	// b(j,k) = P[o(t) | q(t) = j]
	obsModels []model.Modeler

	// Initial state distribution. [nstates x 1]
	// π(i) = P[q(0) = i]; 0<=i<N
	// => saved in log domain.
	initialStateProbs *matrix.Dense

	// Num elements in obs vector.
	numElements int

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
		numElements:       obsModels[0].NumElements(),
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
// 2. Induction:      α(j,t+1) =  sum_{i=0}^{N-1}[α(i,t)a(i,j)] b(j,o(t+1)); 0<=t<T-1; 0<=j<N
// 3. Termination:    P(O/Φ) = sum_{i=0}^{N-1} α(i,T-1)
func (hmm *HMM) alpha(observations *matrix.Dense) (α *matrix.Dense, logProb float64, e error) {

	// Num states.
	N := hmm.nstates

	// expected num rows: numElements
	// expected num cols: T
	ne, T := observations.Dims()

	if ne != hmm.numElements {
		e = fmt.Errorf("Mismatch in num elements in observations [%d] expected [%d].", ne, hmm.numElements)
		return
	}

	// Allocate log-alpha matrix.
	// TODO: use a reusable data structure to minimize garbage.
	α = matrix.MustDense(matrix.ZeroDense(N, T))

	// 1. Initialization. Add in the log doman.
	for i := 0; i < N; i++ {
		α.Set(i, 0, hmm.initialStateProbs.At(i, 0)+hmm.obsModels[i].LogProb(ColumnAt(observations, 0)))
	}

	// 2. Induction.
	for t := 0; t < T-1; t++ {
		for j := 0; j < N; j++ {

			var sum float64
			for i := 0; i < N; i++ {
				sum += math.Exp(α.At(i, t) + hmm.transProbs.At(i, j))
			}
			α.Set(j, t+1, math.Log(sum)+hmm.obsModels[j].LogProb(ColumnAt(observations, t+1)))
		}
	}

	// 3. Termination.
	var sum float64
	for i := 0; i < N; i++ {
		sum += math.Exp(α.At(i, T-1))
	}
	logProb = math.Log(sum)
	return
}

// ColumnAt returns a *matrix.Dense column that is a copy of the values at column c of the matrix.
// Column will panic with ErrIndexOutOfRange is c is not a valid column index.
func ColumnAt(d *matrix.Dense, c int) *matrix.Dense {

	sl := d.Column(c)
	nrows := len(sl)

	col := matrix.MustDense(matrix.ZeroDense(nrows, 1))

	for i := 0; i < nrows; i++ {
		col.Set(i, 0, sl[i])
	}

	return col
}

func (hmm *HMM) NumElements() int {
	return hmm.numElements
}
