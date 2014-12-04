package gaussian2

import (
	"math"
	"math/rand"

	"github.com/akualab/gjoa/floatx"
	"github.com/akualab/gjoa/model"
	"github.com/golang/glog"
	"github.com/gonum/floats"
)

const (
	SMALL_SD        = 0.01
	SMALL_VARIANCE  = SMALL_SD * SMALL_SD
	MIN_NUM_SAMPLES = 0.01
)

// Multivariate Gaussian distribution.
type Gaussian struct {
	ModelName   string    `json:"name,omitempty"`
	NE          int       `json:"num_elements"`
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
}

// Define functions for elementwise transformations.
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

func NewGaussian(numElements int, mean, sd []float64,
	diagonal bool, name string) *Gaussian {

	if !diagonal {
		glog.Fatal("Full covariance matrix is not supported yet.")
	}

	g := EmptyGaussian()
	g.Mean = mean
	g.StdDev = sd
	g.Diag = true
	g.NE = numElements
	g.ModelName = name

	if g.Mean == nil {
		g.Mean = make([]float64, g.NE)
	}

	g.variance = make([]float64, g.NE)
	g.varianceInv = make([]float64, g.NE)
	if g.StdDev == nil {
		g.StdDev = make([]float64, g.NE)
		floatx.Apply(setValueFunc(SMALL_SD), g.StdDev, nil)
	}
	floatx.Sq(g.variance, g.StdDev)

	// Initializes variance, varianceInv, and StdDev.
	g.setVariance(g.variance)

	g.Sumx = make([]float64, g.NE)
	g.Sumxsq = make([]float64, g.NE)

	g.tmpArray = make([]float64, g.NE)
	floatx.Log(g.tmpArray, g.variance)
	g.const1 = -float64(g.NE) * math.Log(2.0*math.Pi) / 2.0
	g.const2 = g.const1 - floats.Sum(g.tmpArray)/2.0

	return g
}

// Returns an empty model with the base modeled initialized.
// Use it reading model from Reader.
func EmptyGaussian() *Gaussian {

	g := &Gaussian{}
	return g
}

// Implements Update() method in Trainer interface.
func (g *Gaussian) Update(x model.Sampler, w func(model.Obs) float64) error {

	c, e := x.Chan()
	if e != nil {
		return e
	}
	for v := range c {
		fv := v.Value().([]float64)
		err := g.UpdateOne(fv, w(v))
		if err != nil {
			return err
		}
	}
	return nil
}

func (g *Gaussian) Predict(x model.Sampler) ([]model.Labeler, error) {

	return nil, nil
}

// Returns log probabilies for samples.
func (g *Gaussian) Score(x model.Sampler) ([]float64, error) {

	c, e := x.Chan()
	if e != nil {
		return nil, e
	}
	scores := make([]float64, 0, 0)
	for v := range c {
		fv := v.Value().([]float64)
		scores = append(scores, g.LogProb(fv))
	}
	return scores, nil
}

// Returns log probability for observation.
func (g *Gaussian) LogProb(obs []float64) (v float64) {

	for i, x := range obs {
		s := g.Mean[i] - x
		v += s * s * g.varianceInv[i] / 2.0
	}
	v = g.const2 - v

	return
}

func (g *Gaussian) Prob(obs []float64) float64 {

	return math.Exp(g.LogProb(obs))
}

func (g *Gaussian) UpdateOne(obs []float64, w float64) error {

	/* Update sufficient statistics. */
	floatx.Apply(scaleFunc(w), obs, g.tmpArray)
	floats.Add(g.Sumx, g.tmpArray)
	floatx.Sq(g.tmpArray, obs)
	floats.Scale(w, g.tmpArray)
	floats.Add(g.Sumxsq, g.tmpArray)
	g.NSamples += w

	return nil
}

// Implements Estimate() method in Trainer interface.
func (g *Gaussian) Estimate() error {

	if g.NSamples > MIN_NUM_SAMPLES {

		/* Estimate the mean. */
		floatx.Apply(scaleFunc(1.0/g.NSamples), g.Sumx, g.Mean)
		/*
		 * Estimate the variance. sigma_sq = 1/n (sumxsq - 1/n sumx^2) or
		 * 1/n sumxsq - mean^2.
		 */
		tmp := g.variance // borrow as an intermediate array.

		//		floatx.Apply(sq, g.Mean, g.tmpArray)
		floatx.Sq(g.tmpArray, g.Mean)
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
	floatx.Log(g.tmpArray, g.variance)
	g.const2 = g.const1 - floats.Sum(g.tmpArray)/2.0

	return nil
}

// Implements Clear() method in Trainer interface.
func (g *Gaussian) Clear() {

	floatx.Apply(setValueFunc(0), g.Sumx, nil)
	floatx.Apply(setValueFunc(0), g.Sumxsq, nil)
	g.NSamples = 0
}

func (g *Gaussian) setVariance(variance []float64) {
	copy(g.variance, variance)
	floatx.Apply(inv, g.variance, g.varianceInv)
	g.StdDev = g.standardDeviation()
}

func (g *Gaussian) standardDeviation() (sd []float64) {

	sd = make([]float64, g.NE)
	floatx.Sqrt(sd, g.variance)
	return
}

func (g *Gaussian) Random(r *rand.Rand) (interface{}, []int, error) {
	obs, e := model.RandNormalVector(g.Mean, g.StdDev, r)
	return obs, nil, e
}

func (g *Gaussian) Name() string        { return g.ModelName }
func (g *Gaussian) NumSamples() float64 { return g.NSamples }
func (g *Gaussian) NumElements() int    { return g.NE }
func (g *Gaussian) SetName(name string) { g.ModelName = name }

func (g *Gaussian) Clone() *Gaussian {

	//	ng = &Gaussian{}

	ng := NewGaussian(g.NE, nil, nil, g.Diag, g.ModelName)

	//	fmt.Printf("xxx ng sumx: %+v\n\n", ng)
	//	fmt.Printf("xxx g  sumx: %+v\n\n", g)

	ng.ModelName = g.ModelName
	ng.NE = g.NE
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
	return ng
}
