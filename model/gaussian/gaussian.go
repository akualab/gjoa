package gaussian

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/akualab/gjoa/floatx"
	"github.com/akualab/gjoa/model"
	"github.com/gonum/floats"
)

const (
	SMALL_SD        = 0.01
	SMALL_VARIANCE  = SMALL_SD * SMALL_SD
	MIN_NUM_SAMPLES = 0.01
)

// Multivariate Gaussian distribution.
type Gaussian struct {
	*model.BaseModel
	ModelName   string    `json:"name,omitempty"`
	NE          int       `json:"num_elements"`
	IsTrainable bool      `json:"trainable"`
	NSamples    float64   `json:"nsamples"`
	Diag        bool      `json:"diag"`
	Sumx        []float64 `json:"sumx,omitempty"`
	Sumxsq      []float64 `json:"sumx_sq,omitempty"`
	Mean        []float64 `json:"mean"`
	StdDev      []float64 `json:"sd"`
	variance    []float64
	varianceInv []float64
	tmpArray    []float64
	const1      float64 // -(N/2)log(2PI) Depends only on NE.
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

func init() {
	m := new(Gaussian)
	model.Register(m)
}

func NewGaussian(numElements int, mean, sd []float64,
	trainable, diagonal bool, name string) (g *Gaussian, e error) {

	if !diagonal {
		e = fmt.Errorf("Full covariance matrix is not supported yet.")
		return
	}

	g = EmptyGaussian()
	g.Mean = mean
	g.StdDev = sd
	g.Diag = true
	g.NE = numElements
	g.ModelName = name
	g.IsTrainable = trainable
	g.fpool = floatx.NewPool(numElements)

	g.Initialize()
	return
}

func (g *Gaussian) Initialize() error {

	if g.Mean == nil {
		g.Mean = make([]float64, g.NE)
	}

	g.variance = make([]float64, g.NE)
	g.varianceInv = make([]float64, g.NE)
	if g.StdDev == nil {
		g.StdDev = make([]float64, g.NE)
		floatx.Apply(setValueFunc(SMALL_SD), g.StdDev, nil)
	}
	floatx.Apply(sq, g.StdDev, g.variance)

	// Initializes variance, varianceInv, and StdDev.
	g.setVariance(g.variance)

	if g.IsTrainable {
		g.Sumx = make([]float64, g.NE)
		g.Sumxsq = make([]float64, g.NE)
	}

	g.tmpArray = make([]float64, g.NE)
	floatx.Apply(log, g.variance, g.tmpArray)
	g.const1 = -float64(g.NE) * math.Log(2.0*math.Pi) / 2.0
	g.const2 = g.const1 - floats.Sum(g.tmpArray)/2.0

	return nil
}

// Returns an empty model with the base modeled initialized.
// Use it reading model from Reader.
func EmptyGaussian() *Gaussian {

	g := &Gaussian{}
	g.BaseModel = model.NewBaseModel(model.Modeler(g))
	return g
}

func (g *Gaussian) LogProb(observation interface{}) (v float64) {

	obs := observation.([]float64)
	for i, x := range obs {
		s := g.Mean[i] - x
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

	if !g.IsTrainable {
		return fmt.Errorf("Attempted to update model [%s] which is not trainable.", g.Name())
	}

	/* Update sufficient statistics. */
	floatx.Apply(scaleFunc(w), obs, g.tmpArray)
	floats.Add(g.Sumx, g.tmpArray)
	floatx.Apply(sq, obs, g.tmpArray)
	floats.Scale(w, g.tmpArray)
	floats.Add(g.Sumxsq, g.tmpArray)
	g.NSamples += w

	return nil
}

func (g *Gaussian) Estimate() error {

	if !g.IsTrainable {
		return fmt.Errorf("Attempted to estimate model [%s] which is not trainable.", g.ModelName)
	}

	if g.NSamples > MIN_NUM_SAMPLES {

		/* Estimate the mean. */
		floatx.Apply(scaleFunc(1.0/g.NSamples), g.Sumx, g.Mean)
		/*
		 * Estimate the variance. sigma_sq = 1/n (sumxsq - 1/n sumx^2) or
		 * 1/n sumxsq - mean^2.
		 */
		tmp := g.variance // borrow as an intermediate array.

		floatx.Apply(sq, g.Mean, g.tmpArray)
		floatx.Apply(scaleFunc(1.0/g.NSamples), g.Sumxsq, tmp)
		floats.SubTo(g.variance, tmp, g.tmpArray)
		floatx.Apply(floorv, g.variance, nil)
	} else {

		/* Not enough training sample. */
		floatx.Apply(setValueFunc(SMALL_VARIANCE), g.variance, nil)
		floatx.Apply(setValueFunc(0), g.Mean, nil)
	}
	g.setVariance(g.variance) // to update varInv and stddev.

	/* Update log Gaussian constant. */
	floatx.Apply(log, g.variance, g.tmpArray)
	g.const2 = g.const1 - floats.Sum(g.tmpArray)/2.0

	return nil
}

func (g *Gaussian) Clear() error {

	if !g.IsTrainable {
		return fmt.Errorf("Attempted to clear model [%s] which is not trainable.", g.ModelName)
	}

	floatx.Apply(setValueFunc(0), g.Sumx, nil)
	floatx.Apply(setValueFunc(0), g.Sumxsq, nil)
	g.NSamples = 0

	return nil
}

func (g *Gaussian) setVariance(variance []float64) {
	copy(g.variance, variance)
	floatx.Apply(inv, g.variance, g.varianceInv)
	g.StdDev = g.standardDeviation()
}

func (g *Gaussian) standardDeviation() (sd []float64) {

	sd = make([]float64, g.NE)
	floatx.Apply(sqrt, g.variance, sd)
	return
}

func (g *Gaussian) Random(r *rand.Rand) (interface{}, []int, error) {
	obs, e := model.RandNormalVector(g.Mean, g.StdDev, r)
	return obs, nil, e
}

func (g *Gaussian) Name() string        { return g.ModelName }
func (g *Gaussian) NumSamples() float64 { return g.NSamples }
func (g *Gaussian) NumElements() int    { return g.NE }
func (g *Gaussian) Trainable() bool     { return g.IsTrainable }
func (g *Gaussian) SetName(name string) { g.ModelName = name }

func (g *Gaussian) Clone() (ng *Gaussian, e error) {

	//	ng = &Gaussian{}

	ng, e = NewGaussian(g.NE, nil, nil, g.IsTrainable, g.Diag, g.ModelName)
	if e != nil {
		return
	}

	ng.Initialize()

	//	fmt.Printf("xxx ng sumx: %+v\n\n", ng)
	//	fmt.Printf("xxx g  sumx: %+v\n\n", g)

	ng.ModelName = g.ModelName
	ng.NE = g.NE
	ng.IsTrainable = g.IsTrainable
	ng.NSamples = g.NSamples
	ng.Diag = g.Diag

	copy(ng.Sumx, g.Sumx)
	copy(ng.Sumxsq, g.Sumxsq)
	copy(ng.Mean, g.Mean)
	copy(ng.StdDev, g.StdDev)
	copy(ng.variance, g.variance)
	copy(ng.varianceInv, g.varianceInv)
	copy(ng.tmpArray, g.tmpArray)
	ng.const1 = g.const1
	ng.const2 = g.const2
	// ng.fpool = g.fpool TODO

	return
}
