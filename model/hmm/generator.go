package hmm

import (
	"fmt"
	"github.com/akualab/gjoa/model"
	"math/rand"
)

// Given the parameters for a HMM and the length of the sequence,
// generates a random sequence from the HMM
func GetRandomVectorFromHMM(
	transProbs [][]float64, initialStateProbs []float64,
	mean []float64, sd []float64, n int) ([]int, []float64, error) {
	r := rand.New(rand.NewSource(1))
	// init
	state0, err0 := model.RandIntFromDist(initialStateProbs, r)
	if err0 != nil {
		return nil, nil, fmt.Errorf("Error calling RandIntFromDist")
	}
	obs0, err00 := model.RandNormalVector(mean, sd, r)
	if err00 != nil {
		return nil, nil, fmt.Errorf("Error calling RandNormalVector")
	}
	obs := make([]float64, n)
	states := make([]int, n)
	obs[0] = obs0[state0]
	states[0] = state0
	for i := 1; i < n; i++ {
		dist := transProbs[state0]
		state, erri := model.RandIntFromDist(dist, r)
		if erri != nil {
			return nil, nil, fmt.Errorf("Error calling RandIntFromDist")
		}
		states[i] = state
		obsi, errii := model.RandNormalVector(mean, sd, r)
		if errii != nil {
			return nil, nil, fmt.Errorf("Error calling RandNormalVector")
		}
		obs[i] = obsi[state]
		state0 = state
	}
	return states, obs, nil
}
