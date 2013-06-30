package model

import ()

// A read-only model.
type Modeler interface {
	Prob() float64
	LogProb() float64
	Name() string
	String() string
	NumElements() int
	Trainable() bool
}

// A trainable model.
type Trainer interface {
	Modeler

	Update(a []float64) error
	Estimate() error
	Clear()
	SetName(name string)
	NumSamples() float64
}
