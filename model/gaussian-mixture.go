package model

import (
	"code.google.com/p/biogo.matrix"
	"fmt"
	"math"
)

const (

	// Estimate mean and variance of Gaussian distribution in the first
	// iteration. Create the target number of Gaussian components
	// (numComponents) in the mixture at the end of the first iteration
	// using the estimated mean and variance.
	GMM_STEP = iota

	// Double the number of Gaussian components at the end of each
	// iteration.
	GMM_DOUBLE

	// Do not allocate structures for training.
	GMM_NONE
)

type GMMConfig struct {
}

type GMM struct {
	model
	diagonal        bool
	trainingMethod  int
	numComponents   int
	posteriorSum    *matrix.Dense
	weights         *matrix.Dense
	logWeights      *matrix.Dense
	tmpProbs1       *matrix.Dense
	tmpProbs2       *matrix.Dense
	totalLikelihood float64
	components      []*Gaussian
	iteration       int
}

// A multivariate Gaussian mixture model.
func NewGaussianMixture(numElements, numComponents int,
	trainable, diagonal bool, name string) (gmm *GMM, e error) {

	if !diagonal {
		e = fmt.Errorf("Full covariance matrix is not supported yet.")
		return
	}

	if !trainable {
		return &GMM{
			numComponents: numComponents,
			diagonal:      true,
			model: model{
				numElements: numElements,
				name:        name,
				trainable:   trainable,
			},
		}, nil
	}

	gmm = &GMM{
		numComponents: numComponents,
		components:    make([]*Gaussian, numComponents, numComponents),
		posteriorSum:  matrix.MustDense(matrix.ZeroDense(numComponents, 1)),
		weights:       matrix.MustDense(matrix.ZeroDense(numComponents, 1)),
		logWeights:    matrix.MustDense(matrix.ZeroDense(numComponents, 1)),
		tmpProbs1:     matrix.MustDense(matrix.ZeroDense(numComponents, 1)),
		tmpProbs2:     matrix.MustDense(matrix.ZeroDense(numComponents, 1)),
		diagonal:      true,
		model: model{
			numElements: numElements,
			name:        name,
			trainable:   trainable,
		},
	}

	for i, _ := range gmm.components {
		cname := getComponentName(name, i, gmm.numComponents)
		gmm.components[i], e = NewGaussian(numElements, nil, nil, trainable, diagonal, cname)
		if e != nil {
			return
		}
	}

	// Initialize weights.
	w := 1.0 / float64(numComponents)
	gmm.weights.ApplyDense(setValueFunc(w), gmm.weights)
	gmm.logWeights.ApplyDense(setValueFunc(math.Log(w)), gmm.logWeights)

	return
}

func (gmm *GMM) Components() []*Gaussian {
	return gmm.components
}

func getComponentName(name string, n, numComponents int) string {

	max := numComponents - 1
	switch {
	case max < 10:
		return fmt.Sprintf("%s-%d", name, n)

	case max < 100:
		return fmt.Sprintf("%s-%02d", name, n)

	case max < 1000:
		return fmt.Sprintf("%s-%03d", name, n)

	case max < 10000:
		return fmt.Sprintf("%s-%04d", name, n)

	default:
		return fmt.Sprintf("%s-%d", name, n)
	}
}
