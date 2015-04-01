// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package model

import (
	"fmt"
	"math/rand"
)

const (
	// DefaultSeed provided for model implementation.
	DefaultSeed = 33
)

// A Modeler type is a complete implementation of a statistical model in gjoa.
type Modeler interface {

	// The model name.
	Name() string

	// Dimensionality of the observation vector.
	Dim() int

	Trainer
	Predictor
	Scorer
	Sampler
}

// A Trainer type can do statictical learning.
type Trainer interface {

	// Updates model using weighted samples: x[i] * w(x[i]).
	Update(x Observer, w func(Obs) float64) error

	// Updates model using a single weighted sample.
	UpdateOne(o Obs, w float64)

	// Estimates model parameters.
	Estimate() error

	// Clears all model parameters.
	Clear()
}

// NoWeight is a parameter for the Update() method.
// Applies a weight of one to the observations.
var NoWeight = func(o Obs) float64 { return 1.0 }

// Weight is a parameter for the Update() method.
// Applies a weight to the observations.
var Weight = func(w float64) func(o Obs) float64 {
	return func(o Obs) float64 {
		return w
	}
}

// Predictor returns a label with a hypothesis given the observations.
type Predictor interface {
	Predict(x Observer) ([]Labeler, error)
}

// Scorer computes log probabilities.
type Scorer interface {
	LogProb(x Obs) float64
}

// The Labeler interface manages data labels.
type Labeler interface {

	// Human-readable name.
	String() string

	// Compare labels.
	IsEqual(label Labeler) bool
}

// Obs is a generic interface to handle observed data.
// Each observation may have a value and a label.
type Obs interface {

	// The observation's id.
	ID() string
	// The observation's value.
	Value() interface{}
	// The observation's label.
	Label() Labeler
}

// The Observer provides streams of observations.
type Observer interface {

	// Returns channel of observations.
	// The sequence ends when the channel closes.
	ObsChan() (<-chan Obs, error)
}

// The Sampler type generates random data using the model.
type Sampler interface {
	// Returns a sample drawn from the underlying distribution.
	Sample(*rand.Rand) Obs

	// Returns a sample of size "size" drawn from the underlying distribution.
	// The sequence ends when the channel closes.
	SampleChan(r *rand.Rand, size int) <-chan Obs
}

// FloatObs implements the Obs interface. Values are slices of type float64.
type FloatObs struct {
	value []float64
	label SimpleLabel
	id    string
}

// NewFloatObs creates new FloatObs objects.
func NewFloatObs(val []float64, lab SimpleLabel) Obs {
	return FloatObs{
		value: val,
		label: lab,
	}
}

// Value method returns the observed value.
func (fo FloatObs) Value() interface{} { return interface{}(fo.value) }

// Label returns the label for the observation.
func (fo FloatObs) Label() Labeler { return Labeler(fo.label) }

// ID returns the observation id.
func (fo FloatObs) ID() string { return fo.id }

// FloatObsSequence implements the Obs interface using a slice of
// float64 slices.
type FloatObsSequence struct {
	value [][]float64
	label SimpleLabel
	id    string
}

// NewFloatObsSequence creates new FloatObsSequence objects.
func NewFloatObsSequence(val [][]float64, lab SimpleLabel, id string) Obs {
	return FloatObsSequence{
		value: val,
		label: lab,
		id:    id,
	}
}

// Value method returns the observed value.
func (fo FloatObsSequence) Value() interface{} { return interface{}(fo.value) }

// ValueAsSlice returns the observed value as a slice of interfaces.
func (fo FloatObsSequence) ValueAsSlice() []interface{} {
	res := make([]interface{}, len(fo.value), len(fo.value))
	for k, v := range fo.value {
		res[k] = v
	}
	return res
}

// Label returns the label for the observation.
func (fo FloatObsSequence) Label() Labeler { return Labeler(fo.label) }

// ID returns the observation id.
func (fo FloatObsSequence) ID() string { return fo.id }

// Add adds a FloatObs to the sequence.
func (fo *FloatObsSequence) Add(obs FloatObs, lab string) {
	fo.value = append(fo.value, obs.value)
	switch {
	case len(lab) > 0 && len(fo.label) == 0:
		x := string(lab) // no sperator
		fo.label = SimpleLabel(x)
	case len(lab) > 0 && len(fo.label) > 0:
		x := string(fo.label) + "," + string(lab)
		fo.label = SimpleLabel(x)
	}
}

// JoinFloatObsSequence joins various FloatObsSequence objects into a new sequence.
// id is the new id of the joined sequence.
func JoinFloatObsSequence(id string, inputs ...*FloatObsSequence) Obs {
	var val [][]float64
	var lab SimpleLabel

	for k, fos := range inputs {
		for _, vec := range fos.value {
			val = append(val, vec)
		}
		if k == 0 {
			lab = fos.label
		} else {
			lab = lab + "," + fos.label
		}
	}

	return &FloatObsSequence{
		value: val,
		label: lab,
		id:    id,
	}
}

// IntObs implements Obs for integer values.
type IntObs struct {
	value int
	label SimpleLabel
	id    string
}

// NewIntObs creates new IntObs objects.
func NewIntObs(val int, lab SimpleLabel, id string) Obs {
	return IntObs{
		value: val,
		label: lab,
		id:    id,
	}
}

// Value method returns the observed value.
func (io IntObs) Value() interface{} { return interface{}(io.value) }

// Label returns the label for the observation.
func (io IntObs) Label() Labeler { return Labeler(io.label) }

// ID returns the observation id.
func (io IntObs) ID() string { return io.id }

// SimpleLabel implements a basic Labeler interface.
type SimpleLabel string

// String returns the label as a string. Multiple labels must be separated using a comma.
func (lab SimpleLabel) String() string {
	//	return lab.name
	return string(lab)
}

// IsEqual compares two labels.
func (lab SimpleLabel) IsEqual(lab2 Labeler) bool {
	if lab.String() == lab2.String() {
		return true
	}
	return false
}

// FloatObserver implements an observer to stream FloatObs objects.
// Not safe to use with multiple goroutines.
type FloatObserver struct {
	Values [][]float64
	Labels []SimpleLabel
	length int
}

// NewFloatObserver creates a new FloatObserver.
func NewFloatObserver(v [][]float64, lab []SimpleLabel) (*FloatObserver, error) {
	if len(v) != len(lab) {
		return nil, fmt.Errorf("length of v [%d] and length of lab [%d] don't match.", len(v), len(lab))
	}
	return &FloatObserver{
		Values: v,
		Labels: lab,
		length: len(v),
	}, nil
}

// ObsChan implements the ObsChan method for the observer interface.
func (fo FloatObserver) ObsChan() (<-chan Obs, error) {

	obsChan := make(chan Obs, 1000)
	go func() {
		for i := 0; i < fo.length; i++ {
			obsChan <- NewFloatObs(fo.Values[i], fo.Labels[i])
		}
		close(obsChan)
	}()

	return obsChan, nil
}

// ObsToF64 converts an Obs to a tuple: []float64, label, id.
func ObsToF64(o Obs) ([]float64, string, string) {
	return o.Value().([]float64), o.Label().String(), o.ID()
}

// F64ToObs converts a []float64 to Obs.
func F64ToObs(v []float64, label string) Obs {
	return NewFloatObs(v, SimpleLabel(label))
}
