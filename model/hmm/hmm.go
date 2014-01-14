/*

Hidden Markov model.


 α β γ ζ Φ

*/
package hmm

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"

	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/floatx"
	"github.com/akualab/gjoa/model"
	"github.com/akualab/gjoa/model/gaussian"
	"github.com/akualab/graph"
	"github.com/golang/glog"
	"github.com/gonum/floats"
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

	if r1 > 0 && r != r1 {
		e = fmt.Errorf("Num states mismatch. transProbs has [%d] and initialStateProbs have [%d].", r, r1)
	}

	// If no initial state probs, use equal probs.
	if r1 == 0 {
		initialStateProbs = make([]float64, r)
		floatx.Apply(setValueFunc(1.0/float64(r)), initialStateProbs, nil)
	}

	// init logTransProbs and logInitProbs
	logTransProbs := floatx.MakeFloat2D(r, r)
	logInitProbs := make([]float64, r)

	// apply log to transProbs and initialStateProbs
	logTransProbs = floatx.Apply2D(log2D, transProbs, logTransProbs)
	logInitProbs = floatx.Apply(log, initialStateProbs, logInitProbs)

	// replace Inf with MaxFloat64 value so we can write/read JSON.
	logTransProbs = floatx.ConvertInfSlice2D(logTransProbs)
	logInitProbs = floatx.ConvertInfSlice(logInitProbs)

	glog.Infof("New HMM. Num states = %d.", r)
	glog.Infof("Init. State Probs:    %v.", initialStateProbs)
	glog.Infof("Log Init Probs:       %v.", logInitProbs)
	glog.Infof("Trans. Probs:         %v.", transProbs)
	glog.Infof("Log Trans. Probs:     %v.", logTransProbs)

	hmm = EmptyHMM()
	hmm.NStates = r
	hmm.TransProbs = logTransProbs
	hmm.ObsModels = obsModels
	hmm.InitProbs = logInitProbs
	hmm.ModelName = name

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

// Returns an empty model with the base modeled initialized.
// Use it reading model from Reader.
func EmptyHMM() *HMM {

	hmm := &HMM{}
	hmm.BaseModel = model.NewBaseModel(model.Modeler(hmm))
	return hmm
}

func (hmm *HMM) Initialize() error {

	if hmm.Config == nil {
		hmm.Config = defaultConfig()
	}
	hmm.generator = NewGenerator(hmm, hmm.Config.HMM.GeneratorSeed)
	return nil
}

func defaultConfig() *gjoa.Config {

	config := &gjoa.Config{
		HMM: gjoa.HMM{
			UpdateIP:        true,
			UpdateTP:        true,
			GeneratorSeed:   0,
			GeneratorMaxLen: 100,
		},
	}
	return config
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

// Returns a map from state model name to state model index.
func (hmm *HMM) Indices() (m map[string]int) {

	m = make(map[string]int)
	for k, v := range hmm.ObsModels {
		m[v.Name()] = k
	}
	return
}

// Returns a map from state model name to state model.
func (hmm *HMM) ModelMap() map[string]model.Modeler {

	m := make(map[string]model.Modeler)
	for name, i := range hmm.Indices() {
		m[name] = hmm.ObsModels[i]
	}
	return m
}

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
	//modelers := make([]model.Modeler, len(part))
	modelers := make([]model.Modeler, 0)

	switch part[0].ModelType {
	case "Gaussian":
		gslice := make([]*gaussian.Gaussian, len(part))
		e = json.Unmarshal(b, &gslice)
		if e != nil {
			return e
		}

		for _, v := range gslice {
			if v == nil {
				glog.Warningf("found null in JSON file for Gaussian - ignoring")
				continue
			}
			v.Initialize()
			//modelers[k] = model.Modeler(v)
			modelers = append(modelers, model.Modeler(v)) // append non-null Gaussians.
		}

	case "GMM":
		gmmslice := make([]*gaussian.GMM, len(part))
		e = json.Unmarshal(b, &gmmslice)
		if e != nil {
			return e
		}

		for k, v := range gmmslice {
			v.Initialize()
			modelers[k] = model.Modeler(v)
		}

	default:
		return fmt.Errorf("Cannot unmarshal json into unknown Modeler type %s.", part[0].ModelType)
	}

	// Assign the slice of Modeler.
	*os = (ObsSlice)(modelers)

	return nil
}

// Write a collection of HMMs to a file.
func WriteHMMCollection(hmms map[string]*HMM, fn string) error {

	f, e := os.Create(fn)
	if e != nil {
		return e
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, v := range hmms {
		glog.V(4).Infof("write hmm %+v", v)
		removeInf(v)
		e := enc.Encode(v)
		if e != nil {
			return e
		}
	}
	return nil
}

// Read a collection of HMMs from a file.
func ReadHMMCollection(fn string) (hmms map[string]*HMM, e error) {

	var f *os.File
	f, e = os.Open(fn)
	if e != nil {
		return
	}
	defer f.Close()
	reader := bufio.NewReader(f)

	hmms = make(map[string]*HMM)

	for {
		var b []byte
		b, e = reader.ReadBytes('\n')
		if e == io.EOF {
			e = nil
			return
		}
		if e != nil {
			return
		}

		hmm := new(HMM)
		e = json.Unmarshal(b, hmm)
		if e != nil {
			return
		}
		hmms[hmm.ModelName] = hmm
	}
	return
}

func removeInf(hmm *HMM) {

	for i, v := range hmm.InitProbs {

		if math.IsInf(v, -1) {
			hmm.InitProbs[i] = -math.MaxFloat64
		}

		for j, w := range hmm.TransProbs[i] {
			if math.IsInf(w, -1) {
				hmm.TransProbs[i][j] = -math.MaxFloat64
			}
		}
	}
}

/* Joins a collection of 2-state HMMs into a single HMM.

   Say we have 2 2-state models: A and B with states A0,A1,B0,B1
   The transition probabilities at the node level are:
     pA00: p     =>    pH00: p
     pA01: 1-p   =>    pH01: 1-p
     pA10: 0     =>    pH10: 0
     pA11: 1     =>    pH11: x
                       pH12: y
                       pH14: z
                       ...

   where x,y,z are transition probabilities in the node graph.
   we find that probability as follows.

   example: src=1, dst=4 (arc from node #1 (second) to node #4 (fifth))
   pH (2 x 1 + 1) (2 x 4) => pH(3,8)

                    +--------------+
   node:   0  0  1  1  2  2  3  3  4  4  5  5  6  6  7  7
   state:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
                    ^              ^
*/
func JoinHMMCollection(g *graph.Graph, hmmColl map[string]*HMM, name string) (hmmOut *HMM) {

	numNodes := len(hmmColl)
	var numStates, index int
	for _, h := range hmmColl {
		numStates += len(h.ObsModels)
	}
	glog.Infof("joining HMM collection with %d models and % states", numNodes, numStates)
	obsModels := make([]model.Modeler, numStates)
	initProbs := make([]float64, numStates)
	probs := floatx.MakeFloat2D(numStates, numStates)
	iProb := 1.0 / float64(numNodes)
	name2Index := make(map[string]int)

	for _, node := range g.GetAll() {

		// Build index.
		name2Index[node.Key()] = index
		// Get model for key.
		m, found := hmmColl[node.Key()]
		if !found {
			glog.Warningf("didn't find model for key %s", node.Key())
			continue
		}

		// Prepare joined data.
		for _, c := range m.ObsModels {
			glog.V(1).Infof("Adding state #%d name: %s", index, c.Name())
			obsModels[index] = c
			probs[index][index] = m.TransProbs[0][0]
			probs[index][index+1] = m.TransProbs[0][1]
			probs[index+1][index] = m.TransProbs[1][0]
			initProbs[index] = iProb
			index += 2
		}
	}

	// The node to node transition probs are done in a second pass because
	// we need the indices of the models in the joined trans prob matrix.
	var sum float64
	for _, node := range g.GetAll() {
		sum = 0
		i := name2Index[node.Key()]
		for succ, p := range node.GetSuccesors() {
			j := name2Index[succ.Key()]
			probs[i+1][j] = p
			sum += p
		}
		if sum > 1 {
			glog.Fatalf("sum greater than 1: [%f]", sum)
		}
		probs[i+1][i+1] = 1 - sum
	}

	config := &gjoa.Config{
		HMM: gjoa.HMM{
			UpdateIP:        false,
			UpdateTP:        false,
			GeneratorSeed:   0,
			GeneratorMaxLen: 100,
		},
	}
	hmmOut, e := NewHMM(probs, initProbs, obsModels, true, name, config)
	gjoa.Fatal(e)
	return
}
