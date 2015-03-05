// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package hmm provides an implementation of hidden Markov models.
It is designed for applications in temporal pattern recognition.

The package can support any output distribution of type model.Modeler.
*/
package hmm

import (
	"math"
	"math/rand"

	"github.com/akualab/gjoa/model"
	"github.com/golang/glog"
)

const (
	smallNumber = 0.000001
	cap         = 1000
)

// ObsSlice type is a slice of modelers.
type ObsSlice []model.Modeler

// Model is a hidden Markov model.
type Model struct {

	// Model name.
	ModelName string `json:"name"`

	// Model Set
	Set *Set `json:"hmm_set"`

	// Train HMM params.
	assigner  Assigner
	generator *Generator
	maxGenLen int
	seed      int64
	updateTP  bool
}

// Option type is used to pass options to NewModel().
type Option func(*Model)

// NewModel creates a new HMM.
func NewModel(options ...Option) *Model {

	m := &Model{
		ModelName: "HMM",
		updateTP:  true,
		seed:      model.DefaultSeed,
		maxGenLen: 100,
	}

	// Set options.
	for _, option := range options {
		option(m)
	}

	//	glog.Infof("New HMM. Num states = %d.", r)

	m.generator = NewGenerator(m)
	return m
}

// UpdateOne updates model using a single weighted sample.
func (model *Model) UpdateOne(o model.Obs, w float64) {

	chain, err := model.Set.chainFromAssigner(o, model.assigner)
	if err != nil {
		glog.Fatalf("failed to update hmm model stats with error: %s", err)
	}
	glog.Info("updating hmm stats for id [%s]", o.ID())
	chain.update()
}

// Update updates sufficient statistics using an observation stream.
func (model *Model) Update(x model.Observer, w func(model.Obs) float64) error {

	//N := hmm.NStates

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

	// tmp := make([]float64, T)
	// for i, g := range γ {
	// 	floatx.Exp(tmp[:T-1], g[:T-1])
	// 	sumg := floats.Sum(tmp[:T-1])
	// 	hmm.SumGamma[i] += sumg

	// 	outputStatePDF := hmm.ObsModels[i].(model.Trainer)
	// 	for t := 0; t < T; t++ {
	// 		o := model.F64ToObs(observations[t])
	// 		outputStatePDF.UpdateOne(o, tmp[t]) // exp(g(t))
	// 	}

	// 	hmm.SumInitProbs[i] += tmp[0] // [3]

	// 	for j, x := range ζ[i] {
	// 		floatx.Exp(tmp[:T-1], x[:T-1])
	// 		hmm.SumXi[i][j] += floats.Sum(tmp[:T-1])
	// 	}
	// }

	//	Sum LogProbs
	//	hmm.SumProb += logProb

	//fmt.Println(α, β, γ, ζ, logProb)
	return nil
}

func (model *Model) Estimate() error {

	// 	// Initial state probabilities.
	// 	s := floats.Sum(hmm.SumInitProbs)
	// 	if hmm.updateIP {
	// 		glog.V(4).Infof("Sum Init. Probs:    %v.", hmm.SumInitProbs)
	// 		floatx.Apply(floatx.ScaleFunc(1.0/s), hmm.SumInitProbs, hmm.InitProbs)
	// 		floatx.Log(hmm.InitProbs, hmm.InitProbs)
	// 	}

	// 	// Transition probabilities.
	// 	if hmm.updateTP {
	// 		for i, sxi := range hmm.SumXi {
	// 			sg := hmm.SumGamma[i]
	// 			floatx.Apply(floatx.ScaleFunc(1.0/sg), sxi, hmm.TransProbs[i])
	// 			floatx.Log(hmm.TransProbs[i], hmm.TransProbs[i])
	// 		}
	// 	}
	// 	for _, m := range hmm.ObsModels {
	// 		m.(model.Trainer).Estimate()
	// 	}
	return nil
	// }

	// func (hmm *Model) Clear() {

	// 	for _, m := range hmm.ObsModels {
	// 		m.(model.Trainer).Clear()
	// 	}
	// 	floatx.Clear2D(hmm.SumXi)
	// 	floatx.Clear(hmm.SumGamma)
	// 	floatx.Clear(hmm.SumInitProbs)
	// 	hmm.SumProb = 0
}

// Returns the log probability.
func (model *Model) LogProb(observation interface{}) float64 {

	// TODO
	//obs := observation.([][]float64)
	return 0
}

// Returns the probability.
func (model *Model) Prob(observation interface{}) float64 {

	obs := observation.([][]float64)
	return math.Exp(model.LogProb(obs))
}

func (model *Model) Random(r *rand.Rand) (interface{}, []int, error) {

	//	return model.generator.Next(model.maxGenLen)
	return nil, nil, nil
}

// Dim is the dimensionality of the observation vector.
func (model *Model) Dim() int {

	// if hmm.ObsModels[0] == nil {
	// 	glog.Fatalf("No observation model available.")
	// }
	// return hmm.ObsModels[0].Dim()
	return 0
}

// func (os *ObsSlice) UnmarshalJSON(b []byte) error {

// 	// We want to peek inside the message to get the model type.
// 	// We copy the bytes to get a raw message first.
// 	bcopy := make([]byte, len(b))
// 	copy(bcopy, b)
// 	rm := json.RawMessage(bcopy)

// 	// Now that we have a raw message we just want to unmarshal the
// 	// json "type" attribute into the ModelType field.
// 	var part []model.BaseModel
// 	e := json.Unmarshal([]byte(rm), &part)
// 	if e != nil {
// 		return e
// 	}

// 	// Couldn't get this to work using reflection so for now I'm using a switch.
// 	// TODO: investigate if we can implement a generic solution using reflection.
// 	//modelers := make([]model.Modeler, len(part))
// 	modelers := make([]model.Modeler, 0)

// 	switch part[0].ModelType {
// 	case "Gaussian":
// 		gslice := make([]*gaussian.Gaussian, len(part))
// 		e = json.Unmarshal(b, &gslice)
// 		if e != nil {
// 			return e
// 		}

// 		for _, v := range gslice {
// 			if v == nil {
// 				glog.Warningf("found null in JSON file for Gaussian - ignoring")
// 				continue
// 			}
// 			v.Initialize()
// 			//modelers[k] = model.Modeler(v)
// 			modelers = append(modelers, model.Modeler(v)) // append non-null Gaussians.
// 		}

// 	case "GMM":
// 		gmmslice := make([]*gaussian.GMM, len(part))
// 		e = json.Unmarshal(b, &gmmslice)
// 		if e != nil {
// 			return e
// 		}

// 		for k, v := range gmmslice {
// 			v.Initialize()
// 			modelers[k] = model.Modeler(v)
// 		}

// 	default:
// 		return fmt.Errorf("Cannot unmarshal json into unknown Modeler type %s.", part[0].ModelType)
// 	}

// 	// Assign the slice of Modeler.
// 	*os = (ObsSlice)(modelers)

// 	return nil
// }

// // Write a collection of HMMs to a file.
// func WriteHMMCollection(hmms map[string]*HMM, fn string) error {

// 	f, e := os.Create(fn)
// 	if e != nil {
// 		return e
// 	}
// 	defer f.Close()
// 	enc := json.NewEncoder(f)
// 	for _, v := range hmms {
// 		glog.V(4).Infof("write hmm %+v", v)
// 		if filterModels(v) {
// 			glog.Warningf("model %s has NaN, removing.", v.ModelName)
// 			continue
// 		}
// 		e := enc.Encode(v)
// 		if e != nil {
// 			return e
// 		}
// 	}
// 	return nil
// }

// // Read a collection of HMMs from a file.
// func ReadHMMCollection(fn string) (hmms map[string]*HMM, e error) {

// 	var f *os.File
// 	f, e = os.Open(fn)
// 	if e != nil {
// 		return
// 	}
// 	defer f.Close()
// 	reader := bufio.NewReader(f)

// 	hmms = make(map[string]*HMM)

// 	for {
// 		var b []byte
// 		b, e = reader.ReadBytes('\n')
// 		if e == io.EOF {
// 			e = nil
// 			return
// 		}
// 		if e != nil {
// 			return
// 		}

// 		hmm := new(HMM)
// 		e = json.Unmarshal(b, hmm)
// 		if e != nil {
// 			return
// 		}
// 		hmms[hmm.ModelName] = hmm
// 	}
// 	return
// }

// // Make models json compatible.
// // Replaces -Inf with -MaxFloat
// // Removes models with NaN
// func filterModels(hmm *HMM) bool {

// 	for i, v := range hmm.InitProbs {

// 		if math.IsInf(v, -1) {
// 			hmm.InitProbs[i] = -math.MaxFloat64
// 		}

// 		if math.IsNaN(v) {
// 			return true
// 		}

// 		for j, w := range hmm.TransProbs[i] {
// 			if math.IsInf(w, -1) {
// 				hmm.TransProbs[i][j] = -math.MaxFloat64
// 			}
// 			if math.IsNaN(w) {
// 				return true
// 			}

// 		}
// 	}
// 	return false
// }

// Name returns the name of the model.
func (model *Model) Name() string {
	return model.ModelName
}

// SetName sets a name for the model.
func (model *Model) setName(name string) {
	model.ModelName = name
}

// Name is an option to set the model name.
func Name(name string) Option {
	return func(model *Model) { model.setName(name) }
}

// Seed sets a seed value for random functions.
// Uses default seed value if omitted.
func Seed(seed int64) Option {
	return func(model *Model) { model.seed = seed }
}

// MaxGenLen option sets the length of the sequences
// created by the MODEL generator. Default is 100.
func MaxGenLen(n int) Option {
	return func(model *Model) { model.maxGenLen = n }
}

// UpdateTP option to update state transition probabilities.
// Default is true.
func UpdateTP(flag bool) Option {
	return func(m *Model) {
		m.updateTP = flag
	}
}

// OSet is an option to set the collection of HMM networks.
func OSet(set *Set) Option {
	return func(model *Model) { model.Set = set }
}

// OAssign is an option to set the label to model assigner.
func OAssign(assigner Assigner) Option {
	return func(model *Model) { model.assigner = assigner }
}
