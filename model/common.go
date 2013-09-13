package model

import (
	"fmt"
	"github.com/gonum/floats"
	"math/rand"
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

func GetRandomVector(mean, std []float64, r *rand.Rand) ([]float64, error) {

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
