package hmm

import (
	"fmt"
	"github.com/akualab/gjoa/model"
	"math/rand"
)

// Generates a random number given a discrete prob distribution.
// This is not optimal but should work for testing
func GetRandomStateFromDist(dist []float64, r *rand.Rand) (int, error) {
	N := len(dist)
	if N == 0 {
		return -1, fmt.Errorf("Error prob distribution has len 0")
	}
	ran := r.Float64()
	cum := 0.0
	for i := 0; i < N; i++ {
		cum = cum + dist[i]
		if ran < cum {
			return i, nil
		}
	}
	if !model.Comparef64(cum, 1.0) {
		return -1, fmt.Errorf("Distribution doesn't sum to 1")
	}
	return N - 1, nil
}

// Given the parameters for a HMM and teh length of the sequence generates
// a random sequence from the HMM
func GetRandomVectorFromHMM(
	transProbs [][]float64, initialStateProbs []float64,
	mean []float64, sd []float64, n int) ([]int, []float64, error) {
	r := rand.New(rand.NewSource(1))
	// init
	state0, err0 := GetRandomStateFromDist(initialStateProbs, r)
	if err0 != nil {
		return nil, nil, fmt.Errorf("Error calling GetRandomStateFromDist")
	}
	obs0, err00 := model.GetRandomVector(mean, sd, r)
	if err00 != nil {
		return nil, nil, fmt.Errorf("Error calling GetRandomVector")
	}
	obs := make([]float64, n)
	states := make([]int, n)
	obs[0] = obs0[state0]
	states[0] = state0
	for i := 1; i < n; i++ {
		dist := transProbs[state0]
		state, erri := GetRandomStateFromDist(dist, r)
		if erri != nil {
			return nil, nil, fmt.Errorf("Error calling GetRandomStateFromDist")
		}
		states[i] = state
		obsi, errii := model.GetRandomVector(mean, sd, r)
		if errii != nil {
			return nil, nil, fmt.Errorf("Error calling GetRandomVector")
		}
		obs[i] = obsi[state]
		state0 = state
	}
	return states, obs, nil
}
