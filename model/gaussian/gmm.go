package gaussian

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/akualab/gjoa/floatx"
	"github.com/akualab/gjoa/model"
	"github.com/golang/glog"
	"github.com/gonum/floats"
)

type GMM struct {
	ModelName    string      `json:"name"`
	NE           int         `json:"num_elements"`
	NSamples     float64     `json:"nsamples"`
	Diag         bool        `json:"diag"`
	NComponents  int         `json:"num_components"`
	PosteriorSum []float64   `json:"posterior_sum,omitempty"`
	Weights      []float64   `json:"-"`
	LogWeights   []float64   `json:"weights,omitempty"`
	Likelihood   float64     `json:"likelihood"`
	Components   []*Gaussian `json:"components,omitempty"`
	Iteration    int         `json:"iteration"`
	tmpProbs     []float64
	rand         *rand.Rand
}

// GMM parameters.
type GMMParam struct {
	NumElements   int
	NumComponents int
	IsFullCov     bool
	Name          string
}

// A multivariate Gaussian mixture model.
//func NewGaussianMixture(numElements, numComponents int,
//	diagonal bool, name string) (gmm *GMM) {
func NewGMM(p GMMParam) *GMM {

	if p.IsFullCov {
		glog.Fatalf("Full covariance matrix is not supported yet.")
	}

	gmm := &GMM{
		NComponents:  p.NumComponents,
		Components:   make([]*Gaussian, p.NumComponents, p.NumComponents),
		PosteriorSum: make([]float64, p.NumComponents),
		LogWeights:   make([]float64, p.NumComponents),
		Diag:         true,
		NE:           p.NumElements,
		ModelName:    p.Name,
		rand:         rand.New(rand.NewSource(seed)),
	}

	for i, _ := range gmm.Components {
		cname := componentName(gmm.ModelName, i, gmm.NComponents)
		gmm.Components[i] = NewGaussian(GaussianParam{
			NumElements: gmm.NE,
			Name:        cname,
			IsFullCov:   !gmm.Diag,
		})

	}

	// Initialize weights.
	logw := -math.Log(float64(gmm.NComponents))
	floatx.Apply(setValueFunc(logw), gmm.LogWeights, nil)

	gmm.tmpProbs = make([]float64, gmm.NComponents)

	// Initialize weights.
	gmm.Weights = make([]float64, gmm.NComponents)
	floatx.Exp(gmm.Weights, gmm.LogWeights)
	return gmm
}

// Computes log prob for mixture.// SIDE EFFECT => returns logProb of
// Gaussian comp + logWeight in matrix pointed by func arg probs.
func (gmm *GMM) logProbInternal(obs, probs []float64) float64 {

	var max float64 = -math.MaxFloat64

	/* Compute log probabilities for this observation. */
	for i, c := range gmm.Components {
		v1 := c.logProb(obs)
		v2 := gmm.LogWeights[i]
		v := v1 + v2

		if probs != nil {
			probs[i] = v
		}
		if v > max {
			max = v
		}
	}

	// To simplify computation, use the max prob in the denominator instead
	// of the sum.
	return max
}

// Returns log probabilies for samples.
func (gmm *GMM) LogProbs(x model.Observer) ([]float64, error) {

	c, e := x.ObsChan()
	if e != nil {
		return nil, e
	}
	scores := make([]float64, 0, 0)
	for v := range c {
		scores = append(scores, gmm.LogProb(v))
	}
	return scores, nil
}

// Returns log probability for observation.
func (gmm *GMM) LogProb(obs model.Obs) float64 {

	o := obs.Value().([]float64)
	return gmm.logProbInternal(o, nil)
}

// Returns the probability.
func (gmm *GMM) prob(obs []float64) float64 {
	return math.Exp(gmm.LogProb(F64ToObs(obs)))
}

func (gmm *GMM) Predict(x model.Observer) ([]model.Labeler, error) {

	glog.Fatal("Predict method not implemented.")
	return nil, nil
}

/*
  The posterior prob for each mixture component. We approximate the sum using max.

                 p(o|c(i)) p(c(i))
   p(c(i)|o) ~ ---------------------
                max{p(o|c(i)) p(c(i))}

*/

func (gmm *GMM) UpdateOne(o model.Obs, w float64) error {

	obs, _ := ObsToF64(o)
	maxProb := gmm.logProbInternal(obs, gmm.tmpProbs)
	gmm.Likelihood += maxProb
	floatx.Apply(addScalarFunc(-maxProb+math.Log(w)), gmm.tmpProbs, nil)

	// Compute posterior probabilities.
	floatx.Exp(gmm.tmpProbs, gmm.tmpProbs)

	// Update posterior sum, needed to compute mixture weights.
	floats.Add(gmm.PosteriorSum, gmm.tmpProbs)

	// Update Gaussian components.
	for i, c := range gmm.Components {
		c.UpdateOne(o, gmm.tmpProbs[i])
	}

	// Count number of observations.
	gmm.NSamples += w

	return nil
}

// Implements Update() method in Trainer interface.
func (gmm *GMM) Update(x model.Observer, w func(model.Obs) float64) error {
	c, e := x.ObsChan()
	if e != nil {
		return e
	}
	for v := range c {
		err := gmm.UpdateOne(v, w(v))
		if err != nil {
			return err
		}
	}
	return nil
}

func (gmm *GMM) Estimate() error {

	// Estimate mixture weights.
	floatx.Apply(scaleFunc(1.0/gmm.NSamples), gmm.PosteriorSum, gmm.Weights)
	floatx.Log(gmm.LogWeights, gmm.Weights)

	// Estimate component density.
	for _, c := range gmm.Components {
		err := c.Estimate()
		if err != nil {
			return err
		}
	}
	gmm.Iteration += 1

	return nil
}

func (gmm *GMM) Clear() {

	for _, c := range gmm.Components {
		c.Clear()
	}
	floatx.Apply(setValueFunc(0), gmm.PosteriorSum, nil)
	gmm.NSamples = 0
	gmm.Likelihood = 0
}

// Returns a random GMM vector
func (gmm *GMM) Sample() model.Obs {
	// Choose a component using weights
	comp, err := model.RandIntFromDist(gmm.Weights, gmm.rand)
	if err != nil {
		glog.Fatalf("Couldn't generate sample. Error: %s", err)
	}
	// Get a random vector from that component
	return gmm.Components[comp].Sample()
}

// SampleChan returns a channel with samples generated by the GMM model.
func (gmm *GMM) SampleChan(size int) <-chan model.Obs {

	if len(gmm.Weights) == 0 {
		glog.Fatal("Parameter Weights is missing.")
	}
	if gmm.rand == nil {
		glog.Fatal("Random value generator is missing.")
	}
	c := make(chan model.Obs, 1000)
	go func() {
		for i := 0; i < size; i++ {
			c <- gmm.Sample()
		}
		close(c)
	}()
	return c
}

// Returns a random vector using the mean and sd vectors.
func RandomVector(mean, sd []float64, r *rand.Rand) []float64 {

	nrows := len(mean)
	if !floats.EqualLengths(mean, sd) {
		panic(floatx.ErrLength)
	}

	vec := make([]float64, nrows)
	for i := 0; i < nrows; i++ {
		v := r.NormFloat64()*sd[i] + mean[i]
		vec[i] = v
	}
	return vec
}

// Generates a random Gaussian mixture model using mean and variance vectors as seed.
// Use this function to initialize the GMM before training. The mean and sd
// vector can be estimated from the data set using a Gaussian model.
func RandomGMM(mean, sd []float64, numComponents int,
	name string, seed int64) *GMM {

	nrows := len(mean)
	if !floats.EqualLengths(mean, sd) {
		panic(floatx.ErrLength)
	}

	gmm := NewGMM(GMMParam{
		NumElements:   nrows,
		NumComponents: numComponents,
		Name:          name,
	})

	r := rand.New(rand.NewSource(seed))
	for _, c := range gmm.Components {
		var rv []float64
		rv = RandomVector(mean, sd, r)
		c.Mean = rv
		variance := make([]float64, len(sd))
		floatx.Sq(variance, sd)
		c.setVariance(variance)
	}
	return gmm
}

// Returns the Gaussian components.
/*
func (gmm *GMM) Components() []*Gaussian {
	return gmm.Components
}
*/
func componentName(name string, n, numComponents int) string {

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

func (gmm *GMM) Name() string        { return gmm.ModelName }
func (gmm *GMM) NumSamples() float64 { return gmm.NSamples }
func (gmm *GMM) NumElements() int    { return gmm.NE }
func (gmm *GMM) SetName(name string) { gmm.ModelName = name }

// Export struct.
/*type GMMValues struct {
	Name          string
	Type          string
	NumElements   int
	NumSamples    float64
	Diagonal      bool
	NumComponents int
	PosteriorSum  []float64 `json:",omitempty"`
	Weights       []float64
	Likelihood    float64
	Components    []interface{}
}

func (gmm *GMM) Values() interface{} {

	values := &GMMValues{
		Name:          gmm.name,
		Type:          gmm.Type(),
		NumElements:   gmm.numElements,
		NumSamples:    gmm.numSamples,
		Diagonal:      gmm.diagonal,
		NumComponents: gmm.numComponents,
		PosteriorSum:  gmm.PosteriorSum,
		Weights:       gmm.Weights,
		Likelihood:    gmm.totalLikelihood,
		Components:    make([]interface{}, gmm.numComponents),
	}

	for k, v := range gmm.components {
		values.Components[k] = v.Values()
	}

	return values
}
*/
/*
func (gmm *GMM) New(values interface{}) (model.Modeler, error) {

	v := values.(*GMMValues)

	ng, e := NewGaussianMixture(v.NumElements, v.NumComponents, true, v.Diagonal, v.Name)
	if e != nil {
		return nil, e
	}

	gaussian := &Gaussian{}
	for k, gv := range v.Components {

		// Create a Gaussian component using the Gaussian values.
		g, e := gaussian.New(gv)
		if e != nil {
			return nil, e
		}
		ng.components[k] = g.(*Gaussian)
	}
	ng.Weights = v.Weights
	ng.numSamples = v.NumSamples
	ng.totalLikelihood = v.Likelihood

	if len(v.PosteriorSum) > 0 {
		ng.PosteriorSum = v.PosteriorSum
	}

	return ng, nil
}
*/
