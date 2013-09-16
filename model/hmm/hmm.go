/*

Hidden Markov model.


 α β γ ζ Φ

*/
package hmm

import (
	"fmt"
	"github.com/akualab/gjoa/floatx"
	"github.com/akualab/gjoa/model"
	"github.com/golang/glog"
	"github.com/gonum/floats"
	"math"
)

const (
	SMALL_NUMBER = 0.000001
)

var exp = func(r int, v float64) float64 { return math.Exp(v) }

func setValueFunc(f float64) floatx.ApplyFunc {
	return func(r int, v float64) float64 { return f }
}

type HMM struct {

	// Model name.
	name string

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
	logTransProbs [][]float64

	// Observation probability distribution functions. [nstates x 1]
	// b(j,k) = P[o(t) | q(t) = j]
	obsModels []model.Trainer

	// Initial state distribution. [nstates x 1]
	// π(i) = P[q(0) = i]; 0<=i<N
	// => saved in log domain.
	logInitProbs []float64

	// Num elements in obs vector.
	numElements int

	// Complete HMM model.
	// Φ = (A, B, π)

	// Train HMM params.
	trainable    bool
	sumXi        [][]float64
	sumGamma     []float64
	sumLogProb   float64
	sumInitProbs []float64
}

// Define functions for elementwise transformations.
var log = func(r int, v float64) float64 { return math.Log(v) }
var log2D = func(r int, c int, v float64) float64 { return math.Log(v) }

// Create a new HMM.
func NewHMM(transProbs [][]float64, initialStateProbs []float64, obsModels []model.Trainer, trainable bool, name string) (hmm *HMM, e error) {

	r, _ := floatx.Check2D(transProbs)
	r1 := len(initialStateProbs)

	if r != r1 {
		e = fmt.Errorf("Num states mismatch. transProbs has [%d] and initialStateProbs have [%d].", r, r1)
	}

	// init logTransProbs and logInitProbs
	logTransProbs := floatx.MakeFloat2D(r, r)
	logInitProbs := make([]float64, r)

	// apply log to transProbs and initialStateProbs
	logTransProbs = floatx.Apply2D(log2D, transProbs, logTransProbs)
	logInitProbs = floatx.Apply(log, initialStateProbs, logInitProbs)

	glog.Infof("New HMM. Num states = %d.", r)
	glog.Infof("Init. State Probs:    %v.", initialStateProbs)
	glog.Infof("Log Init Probs:       %v.", logInitProbs)
	glog.Infof("Trans. Probs:         %v.", transProbs)
	glog.Infof("Log Trans. Probs:     %v.", logTransProbs)

	hmm = &HMM{
		nstates:       r,
		logTransProbs: logTransProbs,
		obsModels:     obsModels,
		logInitProbs:  logInitProbs,
		numElements:   obsModels[0].NumElements(),
		name:          name,
	}

	if !trainable {
		return
	}

	hmm.sumXi = floatx.MakeFloat2D(r, r)
	hmm.sumGamma = make([]float64, r)
	hmm.trainable = true
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
// For scaling details see Rabiner/Juang and
// http://courses.media.mit.edu/2010fall/mas622j/ProblemSets/ps4/tutorial.pdf
func (hmm *HMM) alpha(observations [][]float64) (α [][]float64, logProb float64, e error) {

	// Num states.
	N := hmm.nstates

	// expected num rows: numElements
	// expected num cols: T
	ne, T := floatx.Check2D(observations)

	if ne != hmm.numElements {
		e = fmt.Errorf("Mismatch in num elements in observations [%d] expected [%d].", ne, hmm.numElements)
		return
	}

	if glog.V(3) {
		glog.Infof("N: %d, T: %d", N, T)
	}

	// Allocate log-alpha matrix.
	// TODO: use a reusable data structure to minimize garbage.
	α = floatx.MakeFloat2D(N, T)

	// 1. Initialization. Add in the log domain.
	for i := 0; i < N; i++ {
		α[i][0] = hmm.logInitProbs[i] + hmm.obsModels[i].LogProb(floatx.SubSlice2D(observations, 0))
	}

	// 2. Induction.
	var sumAlphas, sum float64
	for t := 0; t < T-1; t++ {
		sumAlphas = 0
		for j := 0; j < N; j++ {

			sum = 0
			for i := 0; i < N; i++ {
				sum += math.Exp(α[i][t] + hmm.logTransProbs[i][j])
			}
			v := math.Log(sum) + hmm.obsModels[j].LogProb(floatx.SubSlice2D(observations, t+1))
			α[j][t+1] = v

			sumAlphas += math.Exp(v)
			if glog.V(4) {
				glog.Infof("t: %4d | j: %2d | logAlpha: %5e | sumAlphas: %5e", t, j, v, sumAlphas)
			}
		}
		// Applied scale for t independent of j.
		logSumAlphas := math.Log(sumAlphas)
		for j := 0; j < N; j++ {
			v := α[j][t+1] - logSumAlphas
			α[j][t+1] = v
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
func (hmm *HMM) beta(observations [][]float64) (β [][]float64, e error) {

	// Num states.
	N := hmm.nstates

	// expected num rows: numElements
	// expected num cols: T
	ne, T := floatx.Check2D(observations)

	if ne != hmm.numElements {
		e = fmt.Errorf("Mismatch in num elements in observations [%d] expected [%d].", ne, hmm.numElements)
		return
	}

	// Allocate log-beta matrix.
	// TODO: use a reusable data structure to minimize garbage.
	β = floatx.MakeFloat2D(N, T)

	// 1. Initialization. Add in the log doman.
	for i := 0; i < N; i++ {
		β[i][T-1] = 0.0
	}

	// 2. Induction.
	var sumBetas float64
	for t := T - 2; t >= 0; t-- {
		sumBetas = 0
		for i := 0; i < N; i++ {

			var sum float64
			for j := 0; j < N; j++ {

				sum += math.Exp(hmm.logTransProbs[i][j] + // a(i,j)
					hmm.obsModels[j].LogProb(floatx.SubSlice2D(observations, t+1)) + // b(j,o(t+1))
					β[j][t+1]) // β(j,t+1)
			}
			β[i][t] = math.Log(sum)
			sumBetas += sum
		}
		// Applied scale for t independent of i.
		logSumBetas := math.Log(sumBetas)
		for i := 0; i < N; i++ {
			v := β[i][t] - logSumBetas
			β[i][t] = v
		}
	}

	return
}

// Compute gammas. Indices are: γ(state, time)
//
// γ(i,t) =  α(i,t)β(i,t) / sum_{j=0}^{N-1} α(j,t)β(j,t);  0<=j<N
func (hmm *HMM) gamma(α, β [][]float64) (γ [][]float64, e error) {

	αr, αc := floatx.Check2D(α)
	βr, βc := floatx.Check2D(β)

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
	γ = floatx.MakeFloat2D(N, T)

	for t := 0; t < T; t++ {
		var sum float64 = 0.0
		for i := 0; i < N; i++ {
			x := α[i][t] + β[i][t]
			γ[i][t] = x
			sum += math.Exp(x)
		}

		// Normalize.
		for i := 0; i < N; i++ {
			x := γ[i][t] - math.Log(sum)
			γ[i][t] = x
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
func (hmm *HMM) xi(observations, α, β [][]float64) (ζ [][][]float64, e error) {

	a := hmm.logTransProbs
	αr, αc := floatx.Check2D(α)
	βr, βc := floatx.Check2D(β)
	or, oc := floatx.Check2D(observations)

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
	ζ = floatx.MakeFloat3D(N, N, T)

	for t := 0; t < T-1; t++ {
		var sum float64 = 0.0
		for j := 0; j < N; j++ {
			b := hmm.obsModels[j].LogProb(floatx.SubSlice2D(observations, t+1))
			for i := 0; i < N; i++ {
				x := α[i][t] + a[i][j] + b + β[j][t+1]
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
func (hmm *HMM) Update(observations [][]float64, w float64) (e error) {

	if !hmm.trainable {
		return fmt.Errorf("Attempted to train model [%s] which is not trainable.", hmm.name)
	}

	var α, β, γ [][]float64
	var ζ [][][]float64
	var logProb float64

	_, T := floatx.Check2D(observations) // num elements x num obs
	//N := hmm.nstates

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

	// Reestimation of state transition probabilities for one sequence.
	//
	//                   sum_{t=0}^{T-2} ζ(i,j, t)      [1] <== sumXi
	// a_hat(i,j) = ----------------------------------
	//                   sum_{t=0}^{T-2} γ(i,t)         [2] <== sumGamma [without t = T-1]
	//
	//
	// For multiple observation sequences, we need to accumulate counts in the
	// numerator and denominator. Each sequence is normalized using 1/P(k) where
	// P(k) is the P(O/Φ) for the kth observation sequence (this is logProb).

	// Reestimation of initial state probabilities for one sequence.
	// pi+hat(i) = γ(i,0)  [3]  <== sumInitProbs

	// Reestimation of output probability.
	// For state i in sequence k  weigh each observation using  (1/P(k)) sum_{t=0}^{T-2} γ(i,t)

	pk := math.Exp(logProb)
	tmp := make([]float64, T)
	for i, g := range γ {
		floatx.Apply(exp, g[:T-1], tmp[:T-1])
		sumg := floats.Sum(tmp[:T-1]) / pk // [1]
		hmm.sumGamma[i] += sumg

		outputStatePDF := hmm.obsModels[i]
		for t := 0; t < T; t++ {
			obs := floatx.SubSlice2D(observations, t) // TODO: inefficient! REFACTOR: transpose observations matrix everywhere so we don't have to copy slice multiple times. Issue #6
			outputStatePDF.Update(obs, sumg)
		}

		hmm.sumInitProbs[i] += tmp[0] // [3]

		for j, x := range ζ[i] {
			floatx.Apply(exp, x[:T-1], tmp[:T-1])
			hmm.sumXi[i][j] += floats.Sum(tmp[:T-1]) / pk // [2]
		}
	}

	//	Sum LogProbs
	hmm.sumLogProb += logProb

	//fmt.Println(α, β, γ, ζ, logProb)
	return
}

func (hmm *HMM) Estimate() error {

	// Initial state probabilities.
	s := floats.Sum(hmm.sumInitProbs)
	floatx.Apply(floatx.ScaleFunc(1.0/s), hmm.sumInitProbs, hmm.logInitProbs)
	floatx.Apply(floatx.Log, hmm.logInitProbs, nil)

	// Transition probabilities.
	for i, sxi := range hmm.sumXi {
		sg := hmm.sumGamma[i]
		floatx.Apply(floatx.ScaleFunc(1.0/sg), sxi, hmm.logTransProbs[i])
		floatx.Apply(floatx.Log, hmm.logTransProbs[i], nil)
	}
	for _, m := range hmm.obsModels {
		m.Estimate()
	}
	return nil
}

func (hmm *HMM) Clear() error {

	for _, m := range hmm.obsModels {
		m.Clear()
	}
	floatx.Clear2D(hmm.sumXi)
	floatx.Clear(hmm.sumGamma)
	floatx.Clear(hmm.sumInitProbs)
	hmm.sumLogProb = 0

	return nil
}

func (hmm *HMM) SetName(name string) {}
func (hmm *HMM) NumSamples() float64 { return 0.0 }
func (hmm *HMM) NumElements() int    { return hmm.numElements }
func (hmm *HMM) Name() string        { return hmm.name }
