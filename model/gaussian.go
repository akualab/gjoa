package model

import (
	"fmt"
	"github.com/skelterjohn/go.matrix"
)

const (
	SMALL_VARIANCE  = 0.01
	MIN_NUM_SAMPLES = 0.01
)

type Gaussian struct {
	model

	diagonal   bool
	sumx       *matrix.DenseMatrix
	sumxsq     *matrix.DenseMatrix
	numSamples uint64
	mean       *matrix.DenseMatrix
	variance   *matrix.DenseMatrix

	tmpArray matrix.DenseMatrix
	const1   float64 // -(N/2)log(2PI) Depends only on numElements.
	const2   float64 // const1 - sum(log sigma_i) Also depends on variance.
}

func NewGaussian(numElements int, mean, variance matrix.DenseMatrix,
	trainable, diagonal bool, name string) (g *Gaussian, e error) {

	if !diagonal {
		e = fmt.Errorf("Full covariance matrix is not supported yet.")
		return
	}

	g = Gaussian{
		numElements: numElements,
		name:        name,
		trainable:   trainable,
		mean:        mean,
		variance:    variance,
		diagonal:    true,
	}

	return
}

func (g *Gaussian) Mean() *matrix.DenseMatrix {
	return g.sumx
}

func (g *Gaussian) Variance() *matrix.DenseMatrix {
	return g.variance
}
