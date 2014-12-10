package hmm

import (
	"fmt"
	"math/rand"

	"github.com/akualab/gjoa/model"
)

type Generator struct {
	hmm *HMM
	r   *rand.Rand
}

func NewGenerator(hmm *HMM, seed int64) (gen *Generator) {
	r := rand.New(rand.NewSource(seed))
	gen = &Generator{
		hmm: hmm,
		r:   r,
	}
	return gen
}

// Given n, the length of the seq, generates random sequence
// for a given hmm.
func (gen *Generator) Next(n int) ([][]float64, []int, error) {

	obs := make([][]float64, n)
	states := make([]int, n)
	r := gen.r
	logDist := gen.hmm.InitProbs
	state0, err0 := model.RandIntFromLogDist(logDist, r)
	if err0 != nil {
		return nil, nil, fmt.Errorf("Error calling RandIntFromLogDist")
	}
	for i := 0; i < n; i++ {
		states[i] = state0
		g := gen.hmm.ObsModels[state0]

		gs, ok := g.(model.Sampler)
		if !ok {
			return nil, nil, fmt.Errorf("mixture component does not implement the sampler interface")
		}

		x := gs.Sample()
		obs[i] = x.Value().([]float64)
		if err0 != nil {
			return nil, nil, fmt.Errorf("Error generating Random model")
		}
		dist := gen.hmm.TransProbs[state0]
		state, err := model.RandIntFromLogDist(dist, r)
		if err != nil {
			return nil, nil, fmt.Errorf("Error calling RandIntFromLogDist")
		}
		state0 = state
	}
	return obs, states, nil
}
