package model

import (
	"code.google.com/p/biogo.matrix"
	"fmt"
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

func GetRandomVector(mean, std []float64, r *rand.Rand) (*matrix.Dense, error) {

	if len(mean) != len(std) {
		return nil, fmt.Errorf("Cannot generate random vectors length of mean [%d] and std [%d] don't match.",
			len(mean), len(std))
	}
	vector := matrix.MustDense(matrix.ZeroDense(len(mean), 1))
	for i, _ := range mean {
		v := r.NormFloat64()*std[i] + mean[i]
		vector.Set(i, 0, v)
	}

	return vector, nil
}
