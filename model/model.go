package model

import (
	"code.google.com/p/biogo.matrix"
)

// A read-only model.
type Modeler interface {
	Prob(obs *matrix.Dense) float64
	LogProb(obs *matrix.Dense) float64
	Name() string
	//String() string
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
