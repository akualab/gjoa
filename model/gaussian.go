package model

import (
	"code.google.com/p/biogo.matrix"
	"fmt"
	"math"
)

const (
	SMALL_VARIANCE  = 0.01
	MIN_NUM_SAMPLES = 0.01
)

type Gaussian struct {
	model

	diagonal   bool
	sumx       *matrix.Dense
	sumxsq     *matrix.Dense
	numSamples uint64
	mean       *matrix.Dense
	variance   *matrix.Dense
	tmpArray   *matrix.Dense
	const1     float64 // -(N/2)log(2PI) Depends only on numElements.
	const2     float64 // const1 - sum(log sigma_i) Also depends on variance.
}

func NewGaussian(numElements int, mean, variance *matrix.Dense,
	trainable, diagonal bool, name string) (g *Gaussian, e error) {

	if !diagonal {
		e = fmt.Errorf("Full covariance matrix is not supported yet.")
		return
	}

	g = &Gaussian{
		mean:     mean,
		variance: variance,
		diagonal: true,
	}
	g.numElements = numElements
	g.name = name
	g.trainable = trainable

	if mean == nil {
		g.mean = matrix.MustDense(matrix.ZeroDense(numElements, 1))
	}
	if variance == nil {
		sv := func(r, c int, v float64) float64 { return SMALL_VARIANCE }
		g.variance = matrix.MustDense(matrix.ZeroDense(numElements, 1))
		g.variance.ApplyDense(sv, g.variance)
	}
	if trainable {
		g.sumx = matrix.MustDense(matrix.ZeroDense(numElements, 1))
		g.sumxsq = matrix.MustDense(matrix.ZeroDense(numElements, 1))
	}

	log := func(r, c int, v float64) float64 { return math.Log(v) }
	g.tmpArray = matrix.MustDense(matrix.ZeroDense(numElements, 1))
	g.tmpArray = g.variance.ApplyDense(log, g.tmpArray)
	g.const1 = -float64(numElements) * math.Log(2*math.Pi) / 2
	g.const2 = g.const1 - g.tmpArray.Sum()/2.0

	return
}

func (g *Gaussian) Mean() *matrix.Dense {
	return g.sumx
}

func (g *Gaussian) Variance() *matrix.Dense {
	return g.variance
}
