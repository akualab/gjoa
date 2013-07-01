package gjoa

import (
	"bitbucket.org/akualab/gjoa/cache"
)

type FrontEnd struct {
}

type Processer interface {
	Get(frame uint64) []float64
}

type ProcessFunc func(proc *Processor, n uint64) []float64

type Processor struct {
	cache       *cache.Cache
	process     ProcessFunc
	inputs      []Processer
	numElements int
	name        string
}

func NewProcessor(name string, numElements int, cap uint64, proc ProcessFunc, inputs ...Processer) *Processor {

	return &Processor{
		numElements: numElements,
		cache:       cache.NewCache(cap),
		process:     proc,
		inputs:      inputs,
		name:        name,
	}
}

func (p *Processor) Get(n uint64) []float64 {

	v, ok := p.cache.Get(n)
	if ok {
		return v
	}

	frame := p.process(p, n)
	p.cache.Set(n, frame)
	return frame
}

func (p *Processor) Inputs() []Processer {
	return p.inputs
}

func (p *Processor) NumElements() int {
	return p.numElements
}

func (p *Processor) Name() string {
	return p.name
}
