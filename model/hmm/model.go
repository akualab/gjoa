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
	"bytes"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"

	"github.com/akualab/gjoa/model"
	"github.com/akualab/ju"
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
	// Model type.
	Type string `json:"type"`
	// Model name.
	ModelName string `json:"name"`
	// Model Set
	Set *Set `json:"hmm_set"`
	// Train HMM params.
	assigner Assigner
	//	generator *Generator
	maxGenLen     int
	updateTP      bool
	updateOP      bool
	useAlignments bool

	logProb         float64
	updateFailCount int
	updateCount     int
}

// Option type is used to pass options to NewModel().
type Option func(*Model)

// NewModel creates a new HMM.
func NewModel(options ...Option) *Model {

	m := &Model{
		ModelName: "HMM",
		updateTP:  true,
		updateOP:  true,
		maxGenLen: 100,
	}
	m.Type = reflect.TypeOf(*m).String()

	// Set options.
	for _, option := range options {
		option(m)
	}

	//	glog.Infof("New HMM. Num states = %d.", r)

	if m.Set == nil {
		glog.Fatalf("need model set to create HMM model - use the OSet option to specify a model set")
	}
	if m.Set.size() > 1 && m.assigner == nil {
		glog.Fatalf("need assigner to create an HMM model with more than one network - use OAssign option to specify assigner")
	}
	if m.useAlignments {
		m.updateTP = false
	}
	if glog.V(5) {
		glog.Info("created new hmm model")
		glog.Infof("model set: %s", m.Set)
	}
	return m
}

// UpdateOne updates model using a single weighted sample.
func (m *Model) UpdateOne(o model.Obs, w float64) {
	m.updateCount++
	// We create a chain of hmms to update stats for an obs sequence.
	// The chain is released once the update is done.
	// The assigner has the logic for mapping labels to a chain of models.
	chain, err := m.Set.chainFromAssigner(o, m.assigner)
	if err != nil {
		glog.Warningf("skipping, failed to update hmm model stats, oid:%s, error: %s", o.ID(), err)
		return
	}

	// Use the forward-backward algorithm to compute counts.
	if m.useAlignments {
		err = chain.updateFromAlignments()
		if err != nil {
			m.updateFailCount++
			if glog.V(6) {
				glog.Fatal(err)
			}
			glog.Warning(err)
			return
		}
	} else {
		err = chain.update()
		if err != nil {
			m.updateFailCount++
			if glog.V(6) {
				glog.Fatal(err)
			}
			glog.Warning(err)
			return
		}

		// Print log(prob(O/model))\
		p := chain.beta.At(0, 0, 0)
		m.logProb += p
		glog.V(1).Infof("update hmm stats, oid:%s, logProb:%.2f total:%.2f", o.ID(), p, m.logProb)
	}
}

// SetFlags sets various flags in the model.
func (m *Model) SetFlags(useAlignments, updateTP, updateOP bool) {

	m.useAlignments = useAlignments
	m.updateTP = updateTP
	m.updateOP = updateOP

	if m.useAlignments && updateTP {
		glog.Warning("updateTP must be false when useAlignments is true - setting updateTP=false")
		m.updateTP = false
	}
}

// Update updates sufficient statistics using an observation stream.
func (m *Model) Update(x model.Observer, w func(model.Obs) float64) error {
	return nil
}

// Estimate will update HMM parameters from counts.
func (m *Model) Estimate() error {

	glog.Infof("total update counts:%d, update fails:%4.1f%%", m.updateCount, float64(m.updateFailCount*100)/float64(m.updateCount))
	glog.Infof("total logProb:%.2f", m.logProb)

	// Reestimates HMM params in the model set.
	m.Set.reestimate(m.updateTP, m.updateOP)
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
	m.logProb = 0
	m.updateFailCount = 0
	m.updateCount = 0
	m.Set.reset()
}

// ToJSON returns a json string.
func (m *Model) ToJSON() (string, error) {
	var b bytes.Buffer
	err := ju.WriteJSON(&b, m)
	return b.String(), err
}

// String prints the model.
func (m *Model) String() string {
	s, err := m.ToJSON()
	if err != nil {
		panic(err)
	}
	return s
}

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

// UpdateOP option to update output PDF.
// Default is true.
func UpdateOP(flag bool) Option {
	return func(m *Model) {
		m.updateOP = flag
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

// UseAlignments option. When true, the output probability densities are estimated using alignment info.
// The alignment info must be available in the obs object which shoudl implement the model.ALigner interface.
// Will panic if the alignment data is missing.
func UseAlignments(flag bool) Option {
	return func(m *Model) {
		m.useAlignments = flag
	}
}

// IO

// ReadJSON unmarshals json data from an io.Reader anc creates a new HMM model.
func ReadJSON(r io.Reader) (*Model, error) {
	m := NewModel()
	err := ju.ReadJSON(r, m)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// ReadJSONFile unmarshals json data from a file.
func ReadJSONFile(fn string) (*Model, error) {
	f, err := os.Open(fn)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadJSON(f)
}

// WriteJSON writes HMM model to an io.Writer.
func (m *Model) WriteJSON(w io.Writer) error {
	err := ju.WriteJSON(w, m)
	if err != nil {
		return err
	}
	return nil
}

// WriteJSONFile writes to a file.
func (m *Model) WriteJSONFile(fn string) error {
	e := os.MkdirAll(filepath.Dir(fn), 0755)
	if e != nil {
		return e
	}
	f, err := os.Create(fn)
	if err != nil {
		return err
	}
	defer f.Close()
	return m.WriteJSON(f)
}
