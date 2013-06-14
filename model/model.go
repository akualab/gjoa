package model

import ()

// A read-only model.
type ModelRO interface {
	Prob() float64
	LogProb() float64
	Name() string
	String() string
}

// A trainable model.
type Model interface {
	ModelRO

	Update(a []float64) error
	Estimate() error
	Clear()
	SetName(name string)
	NumSamples() float64
}

type model struct {
	name        string
	numElements int
	trainable   bool
	numSamples  float64
}

func (M *model) Name() string        { return M.name }
func (M *model) NumElements() int    { return M.numElements }
func (M *model) Trainable() bool     { return M.trainable }
func (M *model) NumSamples() float64 { return M.numSamples }
