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

	if m.Set == nil {
		glog.Fatalf("cannot create an HMM without a model set - use the OSet option to specify a model set")
	}
	m.generator = NewGenerator(m)
	return m
}

// UpdateOne updates model using a single weighted sample.
func (m *Model) UpdateOne(o model.Obs, w float64) {

	// We create a chain of hmms to update stats for an obs sequence.
	// The chain is released once the update is done.
	// The assigner has the logic for mapping labels to a chain of models.
	chain, err := m.Set.chainFromAssigner(o, m.assigner)
	if err != nil {
		glog.Fatalf("failed to update hmm model stats with error: %s", err)
	}

	// Use the forward-backward algorithm to compute counts.
	chain.update()

	// Print log(prob(O/model))
	glog.Info("update hmm stats, obsid: [%10s], logProb=%10f", o.ID(), chain.beta.At(0, 0, 0))
}

// Update updates sufficient statistics using an observation stream.
func (m *Model) Update(x model.Observer, w func(model.Obs) float64) error {
	return nil
}

// Estimate will update HMM parameters from counts.
func (m *Model) Estimate() error {

	// Reestimates HMM params in the model set.
	m.Set.reestimate()
	return nil
}

// LogProb returns the log probability.
func (m *Model) LogProb(observation interface{}) float64 {

	// TODO
	//obs := observation.([][]float64)
	return 0
}

// Prob returns the probability.
func (m *Model) Prob(observation interface{}) float64 {

	obs := observation.([][]float64)
	return math.Exp(m.LogProb(obs))
}

func (m *Model) Random(r *rand.Rand) (interface{}, []int, error) {

	//	return model.generator.Next(model.maxGenLen)
	return nil, nil, nil
}

// Dim is the dimensionality of the observation vector.
func (m *Model) Dim() int {

	// if hmm.ObsModels[0] == nil {
	// 	glog.Fatalf("No observation model available.")
	// }
	// return hmm.ObsModels[0].Dim()
	return 0
}

// Clear accumulators.
func (m *Model) Clear() {
	m.Set.reset()
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
func (m *Model) Name() string {
	return m.ModelName
}

// SetName sets a name for the model.
func (m *Model) setName(name string) {
	m.ModelName = name
}

// Name is an option to set the model name.
func Name(name string) Option {
	return func(m *Model) { m.setName(name) }
}

// Seed sets a seed value for random functions.
// Uses default seed value if omitted.
func Seed(seed int64) Option {
	return func(m *Model) { m.seed = seed }
}

// MaxGenLen option sets the length of the sequences
// created by the MODEL generator. Default is 100.
func MaxGenLen(n int) Option {
	return func(m *Model) { m.maxGenLen = n }
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
	return func(m *Model) { m.Set = set }
}

// OAssign is an option to set the label to model assigner.
func OAssign(assigner Assigner) Option {
	return func(m *Model) { m.assigner = assigner }
}
