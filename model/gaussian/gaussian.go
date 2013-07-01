package gaussian

import (
	"code.google.com/p/biogo.matrix"
	"fmt"
	"math"
)

const (
	SMALL_VARIANCE  = 0.01
	MIN_NUM_SAMPLES = 0.01
)

// Multivariate Gaussian distribution.
type Gaussian struct {
	name        string
	numElements int
	trainable   bool
	numSamples  float64
	diagonal    bool
	sumx        *matrix.Dense
	sumxsq      *matrix.Dense
	mean        *matrix.Dense
	variance    *matrix.Dense
	varianceInv *matrix.Dense
	tmpArray    *matrix.Dense
	const1      float64 // -(N/2)log(2PI) Depends only on numElements.
	const2      float64 // const1 - sum(log sigma_i) Also depends on variance.
}

// Define functions for elementwise transformations.
var log = func(r, c int, v float64) float64 { return math.Log(v) }
var exp = func(r, c int, v float64) float64 { return math.Exp(v) }
var sq = func(r, c int, v float64) float64 { return v * v }
var sqrt = func(r, c int, v float64) float64 { return math.Sqrt(v) }
var inv = func(r, c int, v float64) float64 { return 1.0 / v }
var floorv = func(r, c int, v float64) float64 {
	if v < SMALL_VARIANCE {
		return SMALL_VARIANCE
	}
	return v
}

func setValueFunc(f float64) matrix.ApplyFunc {
	return func(r, c int, v float64) float64 { return f }
}
func addScalarFunc(f float64) matrix.ApplyFunc {
	return func(r, c int, v float64) float64 { return v + f }
}

func NewGaussian(numElements int, mean, variance *matrix.Dense,
	trainable, diagonal bool, name string) (g *Gaussian, e error) {

	if !diagonal {
		e = fmt.Errorf("Full covariance matrix is not supported yet.")
		return
	}

	g = &Gaussian{
		mean:        mean,
		variance:    variance,
		diagonal:    true,
		numElements: numElements,
		name:        name,
		trainable:   trainable,
	}

	if mean == nil {
		g.mean = matrix.MustDense(matrix.ZeroDense(numElements, 1))
	}
	if variance == nil {
		g.variance = matrix.MustDense(matrix.ZeroDense(numElements, 1))
		g.variance.ApplyDense(setValueFunc(SMALL_VARIANCE), g.variance)
	}
	if trainable {
		g.sumx = matrix.MustDense(matrix.ZeroDense(numElements, 1))
		g.sumxsq = matrix.MustDense(matrix.ZeroDense(numElements, 1))

		g.varianceInv = matrix.MustDense(matrix.ZeroDense(numElements, 1))
		g.variance.ApplyDense(inv, g.varianceInv)
	}

	g.tmpArray = matrix.MustDense(matrix.ZeroDense(numElements, 1))
	g.tmpArray = g.variance.ApplyDense(log, g.tmpArray)
	g.const1 = -float64(numElements) * math.Log(2.0*math.Pi) / 2.0
	g.const2 = g.const1 - g.tmpArray.Sum()/2.0

	return
}

func (g *Gaussian) LogProb(obs *matrix.Dense) float64 {

	g.tmpArray = g.mean.SubDense(obs, g.tmpArray)
	g.tmpArray.ApplyDense(sq, g.tmpArray)
	return g.const2 - g.tmpArray.InnerDense(g.varianceInv)/2.0
}

func (g *Gaussian) Prob(obs *matrix.Dense) float64 {

	return math.Exp(g.LogProb(obs))
}

func (g *Gaussian) Update(obs *matrix.Dense) error {

	if !g.trainable {
		return fmt.Errorf("Attempted to train model [%s] which is not trainable.", g.Name())
	}

	/* Update sufficient statistics. */
	g.sumx.AddDense(obs, g.sumx)
	obs.ApplyDense(sq, g.tmpArray)
	g.tmpArray.AddDense(g.sumxsq, g.sumxsq)
	g.numSamples += 1

	return nil
}

func (g *Gaussian) UpdateSlice(obs []float64) error {

	if g.numElements != len(obs) {
		return fmt.Errorf("Num elements mismatch [%d] vs [%d].", len(obs), g.numElements)
	}
	mat := matrix.MustDense(matrix.ZeroDense(g.numElements, 1))
	for k, v := range obs {
		mat.Set(k, 0, v)
	}
	return g.Update(mat)
}

func (g *Gaussian) WUpdate(obs *matrix.Dense, w float64) error {

	if !g.trainable {
		return fmt.Errorf("Attempted to update model [%s] which is not trainable.", g.Name())
	}

	/* Update sufficient statistics. */
	obs.ScalarDense(w, g.tmpArray)
	g.tmpArray.AddDense(g.sumx, g.sumx)
	obs.ApplyDense(sq, g.tmpArray)
	g.tmpArray.ScalarDense(w, g.tmpArray)
	g.tmpArray.AddDense(g.sumxsq, g.sumxsq)
	g.numSamples += w

	return nil
}

func (g *Gaussian) Estimate() error {

	if !g.trainable {
		return fmt.Errorf("Attempted to estimate model [%s] which is not trainable.", g.name)
	}

	if g.numSamples > MIN_NUM_SAMPLES {

		/* Estimate the mean. */
		g.sumx.ScalarDense(1.0/g.numSamples, g.mean)

		/*
		 * Estimate the variance. sigma_sq = 1/n (sumxsq - 1/n sumx^2) or
		 * 1/n sumxsq - mean^2.
		 */
		tmp := g.variance // borrow as an intermediate array.

		g.mean.ApplyDense(sq, g.tmpArray)
		g.sumxsq.ScalarDense(1.0/g.numSamples, tmp)
		tmp.Sub(g.tmpArray, g.variance)
		g.variance.ApplyDense(floorv, g.variance)
		g.variance.ApplyDense(inv, g.varianceInv)

	} else {

		/* Not enough training sample. */
		g.variance.ApplyDense(setValueFunc(SMALL_VARIANCE), g.variance)
		g.variance.ApplyDense(inv, g.varianceInv)
		g.mean.ApplyDense(setValueFunc(0), g.mean)
	}

	/* Update log Gaussian constant. */
	g.variance.ApplyDense(log, g.tmpArray)
	g.const2 = g.const1 - g.tmpArray.Sum()/2.0

	return nil
}

func (g *Gaussian) Clear() error {

	if !g.trainable {
		return fmt.Errorf("Attempted to clear model [%s] which is not trainable.", g.name)
	}

	g.sumx.ApplyDense(setValueFunc(0), g.sumx)
	g.sumxsq.ApplyDense(setValueFunc(0), g.sumxsq)
	g.numSamples = 0

	return nil
}

func (g *Gaussian) Mean() *matrix.Dense {
	return g.mean
}

func (g *Gaussian) SetMean(mean *matrix.Dense) {
	g.mean = mean
}

func (g *Gaussian) Variance() *matrix.Dense {
	return g.variance
}

func (g *Gaussian) SetVariance(variance *matrix.Dense) {
	g.variance = variance
	g.varianceInv = matrix.MustDense(matrix.ZeroDense(g.numElements, 1))
	g.variance.ApplyDense(inv, g.varianceInv)
}

func (g *Gaussian) StandardDeviation() *matrix.Dense {

	var std *matrix.Dense
	std = g.variance.ApplyDense(sqrt, std)
	return std
}

func (g *Gaussian) Name() string        { return g.name }
func (g *Gaussian) NumSamples() float64 { return g.numSamples }
func (g *Gaussian) NumElements() int    { return g.numElements }
func (g *Gaussian) Trainable() bool     { return g.trainable }
