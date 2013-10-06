package gaussian

import (
	"fmt"
	"github.com/akualab/gjoa/floatx"
	"github.com/akualab/gjoa/model"
	"github.com/gonum/floats"
	"io"
	"math"
	"math/rand"
)

const (
	SMALL_VARIANCE  = 0.01
	MIN_NUM_SAMPLES = 0.01
)

// Multivariate Gaussian distribution.
type Gaussian struct {
	model.Base
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
	fpool       *floatx.Pool
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
		fpool:       floatx.NewPool(numElements),
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
	g.const2 = g.const1 - floats.Sum(g.tmpArray)/2.0

	return
}

func (g *Gaussian) LogProb(observation interface{}) (v float64) {

	obs := observation.([]float64)
	for i, x := range obs {
		s := g.mean[i] - x
		v += s * s * g.varianceInv[i] / 2.0
	}
	v = g.const2 - v

	return
}

func (g *Gaussian) Prob(observation interface{}) float64 {

	obs := observation.([]float64)

	return math.Exp(g.LogProb(obs))
}

func (g *Gaussian) Update(obs []float64, w float64) error {

	if !g.trainable {
		return fmt.Errorf("Attempted to update model [%s] which is not trainable.", g.Name())
	}

	/* Update sufficient statistics. */
	floatx.Apply(scaleFunc(w), obs, g.tmpArray)
	floats.Add(g.sumx, g.tmpArray)
	floatx.Apply(sq, obs, g.tmpArray)
	floats.Scale(w, g.tmpArray)
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
		floatx.Apply(scaleFunc(1.0/g.numSamples), g.sumx, g.mean)
		/*
		 * Estimate the variance. sigma_sq = 1/n (sumxsq - 1/n sumx^2) or
		 * 1/n sumxsq - mean^2.
		 */
		tmp := g.variance // borrow as an intermediate array.

		floatx.Apply(sq, g.mean, g.tmpArray)
		floatx.Apply(scaleFunc(1.0/g.numSamples), g.sumxsq, tmp)
		floats.SubTo(g.variance, tmp, g.tmpArray)
		floatx.Apply(floorv, g.variance, nil)
		floatx.Apply(inv, g.variance, g.varianceInv)
	} else {

		/* Not enough training sample. */
		floatx.Apply(setValueFunc(SMALL_VARIANCE), g.variance, nil)
		floatx.Apply(inv, g.variance, g.varianceInv)
		floatx.Apply(setValueFunc(0), g.mean, nil)
	}

	/* Update log Gaussian constant. */
	floatx.Apply(log, g.variance, g.tmpArray)
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
	floatx.Apply(inv, g.variance, g.varianceInv)
}

func (g *Gaussian) StandardDeviation() (sd []float64) {

	sd = make([]float64, g.numElements)
	floatx.Apply(sqrt, g.variance, sd)
	return
}

func (g *Gaussian) Random(r *rand.Rand) (interface{}, []int, error) {
	obs, e := model.RandNormalVector(g.Mean(), g.StandardDeviation(), r)
	return obs, nil, e
}

func (g *Gaussian) Name() string        { return g.name }
func (g *Gaussian) NumSamples() float64 { return g.numSamples }
func (g *Gaussian) NumElements() int    { return g.numElements }
func (g *Gaussian) Trainable() bool     { return g.trainable }
func (g *Gaussian) SetName(name string) { g.name = name }

// Export struct.
type GaussianValues struct {
	Name        string
	Type        string
	NumElements int
	NumSamples  float64
	Diagonal    bool
	Mean        []float64
	StdDev      []float64
	Sumx        []float64 `json:",omitempty"`
	Sumxsq      []float64 `json:",omitempty"`
}

func (g *Gaussian) Values() interface{} {

	values := &GaussianValues{
		Name:        g.name,
		Type:        "Gaussian",
		NumElements: g.numElements,
		NumSamples:  g.numSamples,
		Diagonal:    g.diagonal,
		Mean:        g.mean,
		StdDev:      g.StandardDeviation(),
		Sumx:        g.sumx,
		Sumxsq:      g.sumxsq,
	}

	return values
}

func (g *Gaussian) Write(w io.Writer) error {

	return g.WriteModel(w, g)
}
