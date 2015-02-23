// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package model

import "fmt"

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
	Name() string

	// Unique id.
	Id() int

	// Compare labels.
	IsEqual(label Labeler) bool
}

// Obs is a generic interface to handle observed data.
// Each observation may have a value and a label.
type Obs interface {

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
	Sample() Obs

	// Returns a sample of size "size" drawn from the underlying distribution.
	// The sequence ends when the channel closes.
	SampleChan(size int) <-chan Obs
}

// FloatObs is an implementation of an Observer whose values are slices of float64.
type FloatObs struct {
	value []float64
	label SimpleLabel
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

// SimpleLabel implements a basic Labeler interface.
type SimpleLabel struct {
	name string
	id   int
}

// Id method returns a unique id for the label.
func (lab SimpleLabel) Id() int {
	return lab.id
}

// Name returns a human-readable label name.
func (lab SimpleLabel) Name() string {
	return lab.name
}

// IsEqual compares two labels.
func (lab SimpleLabel) IsEqual(lab2 Labeler) bool {
	if lab.Id() == lab2.Id() {
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

// ObsToF64 converts an Obs to a tuple with value []float64 and label string.
func ObsToF64(o Obs) ([]float64, string) {
	return o.Value().([]float64), o.Label().Name()
}

// F64ToObs converts a []float64 to Obs.
func F64ToObs(v []float64) Obs {
	return NewFloatObs(v, SimpleLabel{})
}

// ////////////////

// var modelTypes = make(map[string]Modeler)

// // All model types must register during initialization:
// //
// // func init() {
// //	 m := new(MyModel)
// //	 model.Register(m)
// // }
// func Register(model Modeler) {
// 	value := reflect.Indirect(reflect.ValueOf(model))
// 	name := value.Type().Name()
// 	modelTypes[name] = model
// }

// // Returns an uninitialized instance of the model.
// func Model(typeName string) (Modeler, error) {

// 	t := modelTypes[typeName]
// 	if t == nil {
// 		return nil, fmt.Errorf("Unknown model typr [%s].", typeName)
// 	}
// 	return t, nil
// }

// // Returns a random observations generated by the model.
// // obs is the random observation.
// // sequence applies to Markov models.
// type Generator interface {
// 	Random(r *rand.Rand) (obs interface{}, sequence []int, e error)
// }

// // A read-only model.
// type Modeler interface {

// 	// Initializes model. Must be called to initialized private fields when
// 	// the model is created using a Read method.
// 	Initialize() error

// 	// Returns the probabilty of obs given the model.
// 	Prob(obs interface{}) float64

// 	// Returns the log probabilty of obs given the model.
// 	LogProb(obs interface{}) float64

// 	// The model name.
// 	Name() string

// 	// Dimensionality of the observation verctor.
// 	NumElements() int

// 	// True if the model is trainable.
// 	Trainable() bool

// 	Generator
// }

// // A trainable model.
// type Trainer interface {
// 	Modeler

// 	Update(a []float64, w float64) error
// 	Estimate() error
// 	Clear() error
// 	SetName(name string)
// 	NumSamples() float64
// }

// // A trainable sequence model.
// type SequenceTrainer interface {
// 	Modeler

// 	Update(seq [][]float64, w float64) error
// 	Estimate() error
// 	Clear() error
// 	SetName(name string)
// 	NumSamples() float64
// }

// // Implements basic functionality for models. Model implementations can embed
// // this type. The field Base.Model must be initialized to point to the model
// // implementation.
// type BaseModel struct {
// 	Model     Modeler `json:"-"`
// 	ModelType string  `json:"type"`
// }

// func NewBaseModel(model Modeler) *BaseModel {

// 	value := reflect.Indirect(reflect.ValueOf(model))
// 	modelType := value.Type().Name()

// 	return &BaseModel{
// 		Model:     model,
// 		ModelType: modelType,
// 	}
// }

// // Unmarshals data into a struct. The original model instance is not modified.
// // Returns a new model instance of the same type as the original.
// func (base *BaseModel) Read(r io.Reader) (Modeler, error) {

// 	b, err := ioutil.ReadAll(r)
// 	if err != nil {
// 		return nil, err
// 	}

// 	// Get a Modeler object.
// 	model := base.Model
// 	value := reflect.Indirect(reflect.ValueOf(model))
// 	o := reflect.New(value.Type()).Interface().(Modeler)
// 	e := json.Unmarshal(b, &o)

// 	if e != nil {
// 		return nil, e
// 	}
// 	o.Initialize()

// 	return o, nil
// }

// // Reads model data from file. See Read().
// func (base *BaseModel) ReadFile(fn string) (Modeler, error) {

// 	f, err := os.Open(fn)
// 	if err != nil {
// 		return nil, err
// 	}
// 	defer f.Close()
// 	return base.Read(f)
// }

// // Writes model values to an io.Writer.
// func (base *BaseModel) Write(w io.Writer) error {

// 	base.ModelType = base.Type()
// 	b, err := json.Marshal(base.Model)
// 	if err != nil {
// 		return err
// 	}
// 	_, e := w.Write(b)
// 	return e
// }

// // Writes model values to a file.
// func (base *BaseModel) WriteFile(fn string) error {

// 	e := os.MkdirAll(filepath.Dir(fn), 0755)
// 	if e != nil {
// 		return e
// 	}
// 	f, err := os.Create(fn)
// 	if err != nil {
// 		return err
// 	}
// 	defer f.Close()

// 	ee := base.Write(f)
// 	if ee != nil {
// 		return ee
// 	}

// 	glog.Infof("Wrote model \"%s\" to file %s.", base.Model.Name(), fn)
// 	return nil
// }

// // Returns the model type as a string..
// func (base *BaseModel) Type() string {

// 	model := base.Model
// 	value := reflect.Indirect(reflect.ValueOf(model))
// 	name := value.Type().Name()
// 	return name
// }
