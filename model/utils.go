package model

import (
	"fmt"
	"github.com/akualab/gjoa"
	"github.com/gonum/floats"
	"math"
	"math/rand"
)

func RandNormalVector(mean, std []float64, r *rand.Rand) ([]float64, error) {

	if !floats.EqualLengths(mean, std) {
		return nil, fmt.Errorf("Cannot generate random vectors length of mean [%d] and std [%d] don't match.",
			len(mean), len(std))
	}
	vector := make([]float64, len(mean))
	for i, _ := range mean {
		v := r.NormFloat64()*std[i] + mean[i]
		vector[i] = v
	}

	return vector, nil
}

// Generates a random number given a discrete prob distribution.
// This is not optimal but should work for testing
func RandIntFromDist(dist []float64, r *rand.Rand) (int, error) {
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
	if !gjoa.Comparef64(cum, 1.0, 0.001) {
		return -1, fmt.Errorf("Distribution doesn't sum to 1")
	}
	return N - 1, nil
}

// A similar function from above but using log prob.
func RandIntFromLogDist(dist []float64, r *rand.Rand) (int, error) {
	N := len(dist)
	if N == 0 {
		return -1, fmt.Errorf("Error prob distribution has len 0")
	}
	ran := r.Float64()
	cum := 0.0
	for i := 0; i < N; i++ {
		cum = cum + math.Exp(dist[i])
		if ran < cum {
			return i, nil
		}
	}
	if !gjoa.Comparef64(cum, 1.0, 0.001) {
		return -1, fmt.Errorf("Distribution doesn't sum to 1")
	}
	return N - 1, nil
}
