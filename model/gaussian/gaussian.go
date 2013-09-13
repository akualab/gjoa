package gaussian

import (
	"fmt"
	"github.com/akualab/gjoa/floatx"
	"github.com/gonum/floats"
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
	sumx        []float64
	sumxsq      []float64
	mean        []float64
	variance    []float64
	varianceInv []float64
	tmpArray    []float64
	const1      float64 // -(N/2)log(2PI) Depends only on numElements.
	const2      float64 // const1 - sum(log sigma_i) Also depends on variance.
}

// Define functions for elementwise transformations.
var log = func(r int, v float64) float64 { return math.Log(v) }
var exp = func(r int, v float64) float64 { return math.Exp(v) }
var sq = func(r int, v float64) float64 { return v * v }
var sqrt = func(r int, v float64) float64 { return math.Sqrt(v) }
var inv = func(r int, v float64) float64 { return 1.0 / v }
var floorv = func(r int, v float64) float64 {
	if v < SMALL_VARIANCE {
		return SMALL_VARIANCE
	}
	return v
}

func setValueFunc(f float64) floatx.ApplyFunc {
	return func(r int, v float64) float64 { return f }
}
func addScalarFunc(f float64) floatx.ApplyFunc {
	return func(r int, v float64) float64 { return v + f }
}
func scaleFunc(f float64) floatx.ApplyFunc {
	return func(r int, v float64) float64 { return v * f }
}

func NewGaussian(numElements int, mean, variance []float64,
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
		g.mean = make([]float64, numElements)
	}
	if variance == nil {
		g.variance = make([]float64, numElements)
		floatx.Apply(setValueFunc(SMALL_VARIANCE), g.variance, nil)
	}

	g.varianceInv = make([]float64, numElements)
	copy(g.varianceInv, g.variance)
	floatx.Apply(inv, g.varianceInv, nil)

	if trainable {
		g.sumx = make([]float64, numElements)
		g.sumxsq = make([]float64, numElements)
	}

	g.tmpArray = make([]float64, numElements)
	floatx.Apply(log, g.variance, g.tmpArray)
	g.const1 = -float64(numElements) * math.Log(2.0*math.Pi) / 2.0
	//	g.const2 = g.const1 - g.tmpArray.Sum()/2.0
	g.const2 = g.const1 - floats.Sum(g.tmpArray)/2.0

	return
}

func (g *Gaussian) LogProb(obs []float64) float64 {

	//g.tmpArray = g.mean.SubDense(obs, g.tmpArray)
	floats.SubTo(g.tmpArray, g.mean, obs)
	//g.tmpArray.ApplyDense(sq, g.tmpArray)
	floatx.Apply(sq, g.tmpArray, nil)
	//return g.const2 - g.tmpArray.InnerDense(g.varianceInv)/2.0
	return g.const2 - floats.Dot(g.tmpArray, g.varianceInv)/2.0
}

func (g *Gaussian) Prob(obs []float64) float64 {

	return math.Exp(g.LogProb(obs))
}

func (g *Gaussian) Update(obs []float64, w float64) error {

	if !g.trainable {
		return fmt.Errorf("Attempted to update model [%s] which is not trainable.", g.Name())
	}

	/* Update sufficient statistics. */
	//obs.ScalarDense(w, g.tmpArray)
	floatx.Apply(scaleFunc(w), obs, g.tmpArray)
	//g.tmpArray.AddDense(g.sumx, g.sumx)
	floats.Add(g.sumx, g.tmpArray)
	//obs.ApplyDense(sq, g.tmpArray)
	floatx.Apply(sq, obs, g.tmpArray)
	//g.tmpArray.ScalarDense(w, g.tmpArray)
	floats.Scale(w, g.tmpArray)
	//g.tmpArray.AddDense(g.sumxsq, g.sumxsq)
	floats.Add(g.sumxsq, g.tmpArray)
	g.numSamples += w

	return nil
}

func (g *Gaussian) Estimate() error {

	if !g.trainable {
		return fmt.Errorf("Attempted to estimate model [%s] which is not trainable.", g.name)
	}

	if g.numSamples > MIN_NUM_SAMPLES {

		/* Estimate the mean. */
		//g.sumx.ScalarDense(1.0/g.numSamples, g.mean)
		floatx.Apply(scaleFunc(1.0/g.numSamples), g.sumx, g.mean)
		/*
		 * Estimate the variance. sigma_sq = 1/n (sumxsq - 1/n sumx^2) or
		 * 1/n sumxsq - mean^2.
		 */
		tmp := g.variance // borrow as an intermediate array.

		//g.mean.ApplyDense(sq, g.tmpArray)
		floatx.Apply(sq, g.mean, g.tmpArray)
		//g.sumxsq.ScalarDense(1.0/g.numSamples, tmp)
		floatx.Apply(scaleFunc(1.0/g.numSamples), g.sumxsq, tmp)
		//tmp.Sub(g.tmpArray, g.variance)
		floats.SubTo(g.variance, tmp, g.tmpArray)
		//g.variance.ApplyDense(floorv, g.variance)
		floatx.Apply(floorv, g.variance, nil)
		//g.variance.ApplyDense(inv, g.varianceInv)
		floatx.Apply(inv, g.variance, g.varianceInv)

	} else {

		/* Not enough training sample. */
		//g.variance.ApplyDense(setValueFunc(SMALL_VARIANCE), g.variance)
		floatx.Apply(setValueFunc(SMALL_VARIANCE), g.variance, nil)
		//g.variance.ApplyDense(inv, g.varianceInv)
		floatx.Apply(inv, g.variance, g.varianceInv)
		//g.mean.ApplyDense(setValueFunc(0), g.mean)
		floatx.Apply(setValueFunc(0), g.mean, nil)
	}

	/* Update log Gaussian constant. */
	//g.variance.ApplyDense(log, g.tmpArray)
	floatx.Apply(log, g.variance, g.tmpArray)
	//g.const2 = g.const1 - g.tmpArray.Sum()/2.0
	g.const2 = g.const1 - floats.Sum(g.tmpArray)/2.0

	return nil
}

func (g *Gaussian) Clear() error {

	if !g.trainable {
		return fmt.Errorf("Attempted to clear model [%s] which is not trainable.", g.name)
	}

	floatx.Apply(setValueFunc(0), g.sumx, nil)
	floatx.Apply(setValueFunc(0), g.sumxsq, nil)
	g.numSamples = 0

	return nil
}

func (g *Gaussian) Mean() []float64 {
	return g.mean
}

func (g *Gaussian) SetMean(mean []float64) {
	g.mean = mean
}

func (g *Gaussian) Variance() []float64 {
	return g.variance
}

func (g *Gaussian) SetVariance(variance []float64) {
	copy(g.variance, variance)
	g.variance = variance
	copy(g.varianceInv, g.variance)
	floatx.Apply(inv, g.varianceInv, nil)
}

func (g *Gaussian) StandardDeviation() (sd []float64) {

	sd = make([]float64, g.numElements)
	floatx.Apply(sqrt, g.variance, sd)
	return
}

func (g *Gaussian) Name() string        { return g.name }
func (g *Gaussian) NumSamples() float64 { return g.numSamples }
func (g *Gaussian) NumElements() int    { return g.numElements }
func (g *Gaussian) Trainable() bool     { return g.trainable }
