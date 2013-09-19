package model

import (
	"math/rand"
)


// A random generator
type Generator interface {
    Random(r *rand.Rand) ([]float64, error)
}

// A read-only model.
type Modeler interface {
	Prob(obs []float64) float64
	LogProb(obs []float64) float64
	Name() string
	//String() string
	NumElements() int
	Trainable() bool
	Generator
}

// A trainable model.
type Trainer interface {
	Modeler

	Update(a []float64, w float64) error
	Estimate() error
	Clear() error
	SetName(name string)
	NumSamples() float64
}

// A trainable sequence model.
type SequenceTrainer interface {
	Modeler

	Update(seq [][]float64, w float64) error
	Estimate() error
	Clear() error
	SetName(name string)
	NumSamples() float64
}
