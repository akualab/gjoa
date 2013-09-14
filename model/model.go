package model

import ()

// A read-only model.
type Modeler interface {
	Prob(obs []float64) float64
	LogProb(obs []float64) float64
	Name() string
	//String() string
	NumElements() int
	Trainable() bool
}

// A trainable model.
type Trainer interface {
	Modeler

	Update(a []float64, w float64) error
	Estimate() error
	Clear()
	SetName(name string)
	NumSamples() float64
}

// A trainable sequence model.
type SequenceTrainer interface {
	Modeler

	Update(seq [][]float64, w float64) error
	Estimate() error
	Clear()
	SetName(name string)
	NumSamples() float64
}
