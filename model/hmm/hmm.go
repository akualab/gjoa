/*

Hidden Markov model.


 α β γ ζ Φ

*/
package hmm

import (
	"code.google.com/p/biogo.matrix"
	"fmt"
	"github.com/akualab/gjoa/model"
	"github.com/golang/glog"
	"math"
)

const (
	SMALL_NUMBER = 0.000001
)

type HMM struct {

	// q(t) is the state at time t
	// states are labeled {0,2,...,N-1}

	// Number fo hidden states.
	// N
	nstates int

	// State-transition probability distribution matrixi (log scale).
	// [nstates x nstates]
	// a(i,j) = log(P[q(t+1) = j | q(t) = i]); 0 <= i,j <= N-1
	// when transitions between states are not allowed: a(i,j) = 0
	// => saved in log domain.
	logTransProbs *matrix.Dense

	// Observation probability distribution functions. [nstates x 1]
	// b(j,k) = P[o(t) | q(t) = j]
	obsModels []model.Modeler

	// Initial state distribution. [nstates x 1]
	// π(i) = P[q(0) = i]; 0<=i<N
	// => saved in log domain.
	logInitProbs *matrix.Dense

	// Num elements in obs vector.
	numElements int

	// Complete HMM model.
	// Φ = (A, B, π)
}

// Define functions for elementwise transformations.
var log = func(r, c int, v float64) float64 { return math.Log(v) }

// Create a new HMM.
func NewHMM(transProbs, initialStateProbs *matrix.Dense, obsModels []model.Modeler) (hmm *HMM, e error) {

	r, c := transProbs.Dims()
	if r != c {
		e = fmt.Errorf("Matrix transProbs must be square. rows=[%d], cols=[%d]", r, c)
		return
	}

	glog.Infof("New HMM. Num states = [%d].", r)

	// init logTransProbs and logInitProbs
	logTransProbs := matrix.MustDense(matrix.ZeroDense(r, r))
	logInitProbs := matrix.MustDense(matrix.ZeroDense(r, 1))

	// apply log to transProbs and initialStateProbs
	logTransProbs = transProbs.ApplyDense(log, logTransProbs)
	logInitProbs = initialStateProbs.ApplyDense(log, logInitProbs)

	hmm = &HMM{
		nstates:       r,
		logTransProbs: logTransProbs,
		obsModels:     obsModels,
		logInitProbs:  logInitProbs,
		numElements:   obsModels[0].NumElements(),
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
// For scaling details see Rabiner/Juang
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
		α.Set(i, 0, hmm.logInitProbs.At(i, 0)+hmm.obsModels[i].LogProb(ColumnAt(observations, 0)))
	}

	// 2. Induction.
	var sumAlphas, sum float64
	for t := 0; t < T-1; t++ {
		sumAlphas = 0
		for j := 0; j < N; j++ {

			sum = 0
			for i := 0; i < N; i++ {
				sum += math.Exp(α.At(i, t) + hmm.logTransProbs.At(i, j))
			}
			v := math.Log(sum) + hmm.obsModels[j].LogProb(ColumnAt(observations, t+1))
			α.Set(j, t+1, v)

			sumAlphas += math.Exp(v)
			if glog.V(4) {
				glog.Infof("t: %4d | j: %2d | logAlpha: %5e | sumAlphas: %5e", t, j, v, sumAlphas)
			}
		}
		// Applied scale for t independent of j.
		logSumAlphas := math.Log(sumAlphas)
		for j := 0; j < N; j++ {
			v := α.At(j, t+1) - logSumAlphas
			α.Set(j, t+1, v)
		}
		logProb += logSumAlphas
	}

	return
}

// Compute betas. Indices are: β(state, time)
//
// β = | β(0,0),   β(0,1)   ... β(0,T-1)   |
//     | β(1,0),   β(1,1)   ... β(1,T-1)   |
//     ...
//     | β(N-1,0), β(N-1,1) ... β(N-1,T-1) |
//
//
// 1. Initialization: β(i,T-1) = 1;  0<=i<N
// 2. Induction:      β(i,t) =  sum_{j=0}^{N-1} a(i,j) b(j,o(t+1)) β(j,t+1); t=T-2,T-3,...,0; 0<=i<N
func (hmm *HMM) beta(observations *matrix.Dense) (β *matrix.Dense, e error) {

	// Num states.
	N := hmm.nstates

	// expected num rows: numElements
	// expected num cols: T
	ne, T := observations.Dims()

	if ne != hmm.numElements {
		e = fmt.Errorf("Mismatch in num elements in observations [%d] expected [%d].", ne, hmm.numElements)
		return
	}

	// Allocate log-beta matrix.
	// TODO: use a reusable data structure to minimize garbage.
	β = matrix.MustDense(matrix.ZeroDense(N, T))

	// 1. Initialization. Add in the log doman.
	for i := 0; i < N; i++ {
		β.Set(i, T-1, 1.0)
	}

	// 2. Induction.
	var sumBetas float64
	for t := T - 2; t >= 0; t-- {
		sumBetas = 0
		for i := 0; i < N; i++ {

			var sum float64
			for j := 0; j < N; j++ {

				sum += math.Exp(hmm.logTransProbs.At(i, j) + // a(i,j)
					hmm.obsModels[j].LogProb(ColumnAt(observations, t+1)) + // b(j,o(t+1))
					β.At(j, t+1)) // β(j,t+1)
			}
			β.Set(i, t, math.Log(sum))
			sumBetas += sum
		}
		// Applied scale for t independent of i.
		logSumBetas := math.Log(sumBetas)
		for i := 0; i < N; i++ {
			v := β.At(i, t) - logSumBetas
			β.Set(i, t, v)
		}
	}

	return
}

// Compute gammas. Indices are: γ(state, time)
//
// γ(i,t) =  α(i,t)β(i,t) / sum_{j=0}^{N-1} α(j,t)β(j,t);  0<=j<N
func (hmm *HMM) gamma(α, β *matrix.Dense) (γ *matrix.Dense, e error) {

	αr, αc := α.Dims()
	βr, βc := β.Dims()

	if αr != βr || αc != βc {
		e = fmt.Errorf("Shape mismatch: alpha[%d,%d] beta[%d,%d]", αr, αc, βr, βc)
		return
	}

	T := αc
	N := hmm.nstates
	if αr != N {
		e = fmt.Errorf("Num rows [%d] doesn't match num states [%d].", αr, N)
	}

	// Allocate log-gamma matrix.
	// TODO: use a reusable data structure to minimize garbage.
	γ = matrix.MustDense(matrix.ZeroDense(N, T))

	for t := 0; t < T; t++ {
		var sum float64 = 0.0
		for i := 0; i < N; i++ {
			x := α.At(i, t) + β.At(i, t)
			γ.Set(i, t, x)
			sum += math.Exp(x)
		}

		// Normalize.
		for i := 0; i < N; i++ {
			x := γ.At(i, t) - math.Log(sum)
			γ.Set(i, t, x)
		}
	}
	return
}

/*
   Compute xi. Indices are: ζ(from, to, time)

                         α(i,t) a(i,j) b(j,o(t+1)) β(j,t+1)
 ζ(i,j,t) = ------------------------------------------------------------------
            sum_{i=0}^{N-1} sum_{j=0}^{N-1} α(i,t) a(i,j) b(j,o(t+1)) β(j,t+1)

*/
func (hmm *HMM) xi(observations, α, β *matrix.Dense) (ζ [][][]float64, e error) {

	a := hmm.logTransProbs
	αr, αc := α.Dims()
	βr, βc := β.Dims()
	or, oc := observations.Dims()

	if αr != βr || αc != βc {
		e = fmt.Errorf("Shape mismatch: alpha[%d,%d] beta[%d,%d]", αr, αc, βr, βc)
		return
	}

	T := αc
	N := hmm.nstates
	if or != hmm.numElements {
		e = fmt.Errorf("Mismatch in num elements in observations [%d] expected [%d].",
			or, hmm.numElements)
		return
	}
	if oc != T {
		e = fmt.Errorf("Mismatch in T observations has [%d], expected [%d].", oc, T)
		return
	}
	if αr != N {
		e = fmt.Errorf("Num rows [%d] doesn't match num states [%d].", αr, N)
	}

	// Allocate log-xi matrix.
	// TODO: use a reusable data structure to minimize garbage.
	ζ = make([][][]float64, N)
	for i := 0; i < N; i++ {
		ζ[i] = make([][]float64, N)
		for j := 0; j < N; j++ {
			for t := 0; t < N; t++ {
				ζ[i][j] = make([]float64, T)
			}
		}
	}

	for t := 0; t < T-1; t++ {
		var sum float64 = 0.0
		for j := 0; j < N; j++ {
			b := hmm.obsModels[j].LogProb(ColumnAt(observations, t+1))
			for i := 0; i < N; i++ {
				x := α.At(i, t) + a.At(i, j) + b + β.At(j, t+1)
				ζ[i][j][t] = x
				sum += math.Exp(x)
			}
		}
		// Normalize.
		for i := 0; i < N; i++ {
			for j := 0; j < N; j++ {
				ζ[i][j][t] -= math.Log(sum)
			}
		}
	}
	return
}

// Update model statistics.
// sequence is a matrix
func (hmm *HMM) Update(observations *matrix.Dense) (e error) {

	var α, β, γ *matrix.Dense
	var ζ [][][]float64
	var logProb float64

	// Compute  α, β, γ, ζ
	// TODO: compute α, β, concurrently using go routines.
	α, logProb, e = hmm.alpha(observations)
	if e != nil {
		return
	}
	β, e = hmm.beta(observations)
	if e != nil {
		return
	}
	// TODO: compute γ, ζ concurrently using go routines.
	// Can we compute ζ more efficiently using γ?
	γ, e = hmm.gamma(α, β)
	if e != nil {
		return
	}
	ζ, e = hmm.xi(observations, α, β)
	if e != nil {
		return
	}

	fmt.Println(α, β, γ, ζ, logProb)
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
