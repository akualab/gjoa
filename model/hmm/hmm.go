/*

Hidden Markov model.


 α β γ ζ Φ

*/
package hmm

import (
	"encoding/json"
	"fmt"
	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/floatx"
	"github.com/akualab/gjoa/model"
	"github.com/akualab/gjoa/model/gaussian"
	"github.com/golang/glog"
	"github.com/gonum/floats"
	"math"
	"math/rand"
)

const (
	SMALL_NUMBER = 0.000001
)

var exp = func(r int, v float64) float64 { return math.Exp(v) }

func setValueFunc(f float64) floatx.ApplyFunc {
	return func(r int, v float64) float64 { return f }
}

type ObsSlice []model.Modeler

type HMM struct {
	*model.BaseModel

	// Model name.
	ModelName string `json:"name"`

	// q(t) is the state at time t
	// states are labeled {0,2,...,N-1}

	// Number fo hidden states.
	// N
	NStates int `json:"num_states"`

	// State-transition probability distribution matrixi (log scale).
	// [nstates x nstates]
	// a(i,j) = log(P[q(t+1) = j | q(t) = i]); 0 <= i,j <= N-1
	// when transitions between states are not allowed: a(i,j) = 0
	// => saved in log domain.
	TransProbs [][]float64 `json:"trans_probs"`

	// Initial state distribution. [nstates x 1]
	// π(i) = P[q(0) = i]; 0<=i<N
	// => saved in log domain.
	InitProbs []float64 `json:"init_probs,omitempty"`

	// Observation probability distribution functions. [nstates x 1]
	// b(j,k) = P[o(t) | q(t) = j]
	ObsModels ObsSlice `json:"obs_models,omitempty"`

	// Complete HMM model.
	// Φ = (A, B, π)

	// Train HMM params.
	IsTrainable  bool        `json:"trainable"`
	SumXi        [][]float64 `json:"sum_xi,omitempty"`
	SumGamma     []float64   `json:"sum_gamma,omitempty"`
	SumProb      float64     `json:"sum_probs,omitempty"`
	SumInitProbs []float64   `json:"sum_init_probs,omitempty"`
	generator    *Generator
	Config       *gjoa.Config `json:"trainer_config,omitempty"`
}

// Define functions for elementwise transformations.
var log = func(r int, v float64) float64 { return math.Log(v) }
var log2D = func(r int, c int, v float64) float64 { return math.Log(v) }

func init() {
	m := new(HMM)
	model.Register(m)
}

// Create a new HMM.
func NewHMM(transProbs [][]float64, initialStateProbs []float64, obsModels []model.Modeler, trainable bool, name string, config *gjoa.Config) (hmm *HMM, e error) {

	r, _ := floatx.Check2D(transProbs)
	r1 := len(initialStateProbs)

	if r != r1 {
		e = fmt.Errorf("Num states mismatch. transProbs has [%d] and initialStateProbs have [%d].", r, r1)
	}

	// Set default config values.
	if config == nil {

		config = &gjoa.Config{
			HMM: gjoa.HMM{
				UpdateIP:        true,
				UpdateTP:        true,
				GeneratorSeed:   0,
				GeneratorMaxLen: 100,
			},
		}
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
		NStates:    r,
		TransProbs: logTransProbs,
		ObsModels:  obsModels,
		InitProbs:  logInitProbs,
		ModelName:  name,
		Config:     config,
	}

	// Initialize base model.
	hmm.BaseModel = model.NewBaseModel(hmm)

	if !trainable {
		goto INIT
	}

	hmm.SumXi = floatx.MakeFloat2D(r, r)
	hmm.SumGamma = make([]float64, r)
	hmm.SumInitProbs = make([]float64, r)
	hmm.IsTrainable = true

INIT:
	e = hmm.Initialize()
	if e != nil {
		return
	}
	return
}

func (hmm *HMM) Initialize() error {

	hmm.generator = NewGenerator(hmm, hmm.Config.HMM.GeneratorSeed)
	return nil
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
	N := hmm.NStates

	// num rows: T
	// num cols: ne
	T, _ := floatx.Check2D(observations)

	// Allocate log-alpha matrix.
	// TODO: use a reusable data structure to minimize garbage.
	α = floatx.MakeFloat2D(N, T)

	// 1. Initialization. Add in the log domain.
	for i := 0; i < N; i++ {
		α[i][0] = hmm.InitProbs[i] + hmm.ObsModels[i].LogProb(observations[0])
	}

	// 2. Induction.
	var sumAlphas, sum float64
	for t := 0; t < T-1; t++ {
		sumAlphas = 0
		for j := 0; j < N; j++ {

			sum = 0
			for i := 0; i < N; i++ {
				sum += math.Exp(α[i][t] + hmm.TransProbs[i][j])
			}
			v := math.Log(sum) + hmm.ObsModels[j].LogProb(observations[t+1])
			α[j][t+1] = v

			sumAlphas += math.Exp(v)
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
	N := hmm.NStates

	// num rows: T
	// num cols: numElements
	T, _ := floatx.Check2D(observations)

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

				sum += math.Exp(hmm.TransProbs[i][j] + // a(i,j)
					hmm.ObsModels[j].LogProb(observations[t+1]) + // b(j,o(t+1))
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
// γ(i,t) =  α(i,t)β(i,t) / sum_{j=0}^{N-1} α(j,t)β(i,t);  0<=j<N
func (hmm *HMM) gamma(α, β [][]float64) (γ [][]float64, e error) {

	αr, αc := floatx.Check2D(α)
	βr, βc := floatx.Check2D(β)

	if αr != βr || αc != βc {
		e = fmt.Errorf("Shape mismatch: alpha[%d,%d] beta[%d,%d]", αr, αc, βr, βc)
		return
	}

	T := αc
	N := hmm.NStates
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

	a := hmm.TransProbs
	αr, αc := floatx.Check2D(α)
	βr, βc := floatx.Check2D(β)
	oc, _ := floatx.Check2D(observations)

	if αr != βr || αc != βc {
		e = fmt.Errorf("Shape mismatch: alpha[%d,%d] beta[%d,%d]", αr, αc, βr, βc)
		return
	}

	T := αc
	N := hmm.NStates
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
			b := hmm.ObsModels[j].LogProb(observations[t+1])
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

	if !hmm.IsTrainable {
		return fmt.Errorf("Attempted to train model [%s] which is not trainable.", hmm.ModelName)
	}

	var α, β, γ [][]float64
	var ζ [][][]float64
	var logProb float64

	T, _ := floatx.Check2D(observations) // num elements x num obs
	//N := hmm.NStates

	// Compute  α, β, γ, ζ
	α, β, logProb, e = hmm.alphaBeta(observations)
	if e != nil {
		return
	}
	γ, ζ, e = hmm.gammaXi(observations, α, β)
	if e != nil {
		return
	}

	/*
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
	*/
	// Reestimation of state transition probabilities for one sequence.
	//
	//                   sum_{t=0}^{T-2} ζ(i,j, t)      [1] <== SumXi
	// a_hat(i,j) = ----------------------------------
	//                   sum_{t=0}^{T-2} γ(i,t)         [2] <== SumGamma [without t = T-1]
	//
	//

	// Reestimation of initial state probabilities for one sequence.
	// pi+hat(i) = γ(i,0)  [3]  <== SumInitProbs

	// Reestimation of output probability.
	// For state i in sequence k  weigh each observation using
	// sum_{t=0}^{T-2} γ(i,t)

	tmp := make([]float64, T)
	for i, g := range γ {
		floatx.Apply(exp, g[:T-1], tmp[:T-1])
		sumg := floats.Sum(tmp[:T-1])
		hmm.SumGamma[i] += sumg

		outputStatePDF := hmm.ObsModels[i].(model.Trainer)
		for t := 0; t < T; t++ {
			obs := observations[t]
			outputStatePDF.Update(obs, tmp[t]) // exp(g(t))
		}

		hmm.SumInitProbs[i] += tmp[0] // [3]

		for j, x := range ζ[i] {
			floatx.Apply(exp, x[:T-1], tmp[:T-1])
			hmm.SumXi[i][j] += floats.Sum(tmp[:T-1])
		}
	}

	//	Sum LogProbs
	hmm.SumProb += logProb

	//fmt.Println(α, β, γ, ζ, logProb)
	return
}

func (hmm *HMM) Estimate() error {

	// Initial state probabilities.
	s := floats.Sum(hmm.SumInitProbs)
	if hmm.Config.HMM.UpdateIP {
		glog.Infof("Sum Init. Probs:    %v.", hmm.SumInitProbs)
		floatx.Apply(floatx.ScaleFunc(1.0/s), hmm.SumInitProbs, hmm.InitProbs)
		floatx.Apply(floatx.Log, hmm.InitProbs, nil)
	}

	// Transition probabilities.
	if hmm.Config.HMM.UpdateTP {
		for i, sxi := range hmm.SumXi {
			sg := hmm.SumGamma[i]
			floatx.Apply(floatx.ScaleFunc(1.0/sg), sxi, hmm.TransProbs[i])
			floatx.Apply(floatx.Log, hmm.TransProbs[i], nil)
		}
	}
	for _, m := range hmm.ObsModels {
		m.(model.Trainer).Estimate()
	}
	return nil
}

func (hmm *HMM) Clear() error {

	for _, m := range hmm.ObsModels {
		m.(model.Trainer).Clear()
	}
	floatx.Clear2D(hmm.SumXi)
	floatx.Clear(hmm.SumGamma)
	floatx.Clear(hmm.SumInitProbs)
	hmm.SumProb = 0

	return nil
}

// Returns the log probability.
func (hmm *HMM) LogProb(observation interface{}) float64 {

	// TODO
	//obs := observation.([][]float64)
	return 0
}

// Returns the probability.
func (hmm *HMM) Prob(observation interface{}) float64 {

	obs := observation.([][]float64)
	return math.Exp(hmm.LogProb(obs))
}

func (hmm *HMM) Random(r *rand.Rand) (interface{}, []int, error) {

	return hmm.generator.Next(hmm.Config.HMM.GeneratorMaxLen)
}

func (hmm *HMM) SetName(name string) {}
func (hmm *HMM) NumSamples() float64 { return 0.0 }
func (hmm *HMM) NumElements() int {

	if hmm.ObsModels[0] == nil {
		glog.Fatalf("No observation model available.")
	}
	return hmm.ObsModels[0].NumElements()
}
func (hmm *HMM) Name() string    { return hmm.ModelName }
func (hmm *HMM) Trainable() bool { return hmm.IsTrainable }

// Compute α and β.
func (hmm *HMM) alphaBeta(observations [][]float64) (α, β [][]float64, logProb float64, e error) {

	α, logProb, e = hmm.alpha(observations)
	if e != nil {
		return
	}
	β, e = hmm.beta(observations)
	if e != nil {
		return
	}

	return
}

// Compute α and β concurrently.
func (hmm *HMM) concurrentAlphaBeta(observations [][]float64) (α, β [][]float64, logProb float64, e error) {

	α_done := make(chan bool)
	β_done := make(chan bool)
	//t0 := time.Now()

	// Launch in separate go routines.
	go func() {
		α, logProb, e = hmm.alpha(observations)
		if e != nil {
			return
		}
		α_done <- true
	}()
	go func() {
		β, e = hmm.beta(observations)
		if e != nil {
			return
		}
		β_done <- true
	}()

	// Wait for both α and β to finish.
	for i := 0; i < 2; i++ {
		select {
		case <-α_done:
			//glog.V(5).Infof("Alpha took: %v", time.Now().Sub(t0))
		case <-β_done:
			//glog.V(5).Infof("Beta took: %v", time.Now().Sub(t0))
		}
	}
	//glog.V(5).Infof("Alpha+Beta took: %v", time.Now().Sub(t0))

	return
}

// Compute γ and ζ.
func (hmm *HMM) gammaXi(observations, α, β [][]float64) (γ [][]float64, ζ [][][]float64, e error) {

	γ, e = hmm.gamma(α, β)
	if e != nil {
		return
	}
	ζ, e = hmm.xi(observations, α, β)
	if e != nil {
		return
	}

	return
}

// Compute γ and ζ concurrently.
func (hmm *HMM) concurrentGammaXi(observations, α, β [][]float64) (γ [][]float64, ζ [][][]float64, e error) {

	γ_done := make(chan bool)
	ζ_done := make(chan bool)

	// Launch in separate go routines.
	go func() {
		γ, e = hmm.gamma(α, β)
		if e != nil {
			return
		}
		γ_done <- true
	}()
	go func() {
		ζ, e = hmm.xi(observations, α, β)
		if e != nil {
			return
		}
		ζ_done <- true
	}()

	// Wait for both γ and ζ to finish.
	for i := 0; i < 2; i++ {
		select {
		case <-γ_done:
		case <-ζ_done:
		}
	}

	return
}

func compareSliceFloat(s1, s2 []float64) bool {
	for i, _ := range s1 {
		if !model.Comparef64(s1[i], s2[i], 0.0001) {
			fmt.Errorf("s1[%d]: %f, s2[%d]: %f", s1, s2)
			return false
		}
	}
	return true
}

type Models struct {
	ModelTypes []model.BaseModel
}

func (os *ObsSlice) UnmarshalJSON(b []byte) error {

	// We want to peek inside the message to get the model type.
	// We copy the bytes to get a raw message first.
	bcopy := make([]byte, len(b))
	copy(bcopy, b)
	rm := json.RawMessage(bcopy)

	// Now that we have a raw message we just want to unmarshal the
	// json "type" attribute into the ModelType field.
	var part []model.BaseModel
	e := json.Unmarshal([]byte(rm), &part)
	if e != nil {
		return e
	}

	// Couldn't get this to work using reflection so for now I'm using a switch.
	// TODO: investigate if we can implement a generic solution using reflection.
	modelers := make([]model.Modeler, len(part))

	switch part[0].ModelType {
	case "Gaussian":
		gslice := make([]*gaussian.Gaussian, len(part))
		e = json.Unmarshal(b, &gslice)
		if e != nil {
			return e
		}

		for k, v := range gslice {
			modelers[k] = model.Modeler(v)
		}

	case "GMM":
		gmmslice := make([]*gaussian.GMM, len(part))
		e = json.Unmarshal(b, &gmmslice)
		if e != nil {
			return e
		}

		for k, v := range gmmslice {
			modelers[k] = model.Modeler(v)
		}

	default:
		return fmt.Errorf("Cannot unmarshal json into unknown Modeler type %s.", part[0].ModelType)
	}

	// Assign the slice of Modeler.
	*os = (ObsSlice)(modelers)

	return nil
}
