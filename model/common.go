package model

import (
	"fmt"
	"github.com/gonum/floats"
	"math"
	"math/rand"
	"testing"
)

const epsilon = 0.004

func Comparef64(f1, f2 float64) bool {
	err := f2 - f1
	if err < 0 {
		err = -err
	}
	if err < epsilon {
		return true
	}
	return false
}

func CompareSliceFloat(t *testing.T, expected []float64, actual []float64, message string) {
	for i, _ := range expected {
		if !Comparef64(expected[i], actual[i]) {
			t.Errorf("[%s]. Expected: [%f], Got: [%f]",
				message, expected[i], actual[i])
		}
	}
}

func CompareFloats(t *testing.T, expected float64, actual float64, message string) {
	if !Comparef64(expected, actual) {
		t.Errorf("[%s]. Expected: [%f], Got: [%f]",
			message, expected, actual)
	}
}

func CompareSliceInt(t *testing.T, expected []int, actual []int, message string) {
	for i, _ := range expected {
		if expected[i] != actual[i] {
			t.Errorf("[%s]. Expected: [%d], Got: [%d]",
				message, expected[i], actual[i])
		}
	}
}

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
	if !Comparef64(cum, 1.0) {
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
	if !Comparef64(cum, 1.0) {
		return -1, fmt.Errorf("Distribution doesn't sum to 1")
	}
	return N - 1, nil
}
