package gaussian

import (
	"fmt"
	"github.com/akualab/gjoa/floatx"
	"github.com/akualab/gjoa/model"
	"github.com/gonum/floats"
	"math"
	"math/rand"
)

type GMMConfig struct {
}

type GMM struct {
	*model.BaseModel
	ModelName    string      `json:"name"`
	NE           int         `json:"num_elements"`
	IsTrainable  bool        `json:"trainable"`
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
}

func init() {
	m := new(GMM)
	model.Register(m)
}

// A multivariate Gaussian mixture model.
func NewGaussianMixture(numElements, numComponents int,
	trainable, diagonal bool, name string) (gmm *GMM, e error) {

	if !diagonal {
		e = fmt.Errorf("Full covariance matrix is not supported yet.")
		return
	}

	if !trainable {
		gmm = &GMM{
			NComponents: numComponents,
			Diag:        true,
			NE:          numElements,
			ModelName:   name,
			IsTrainable: trainable,
		}
		return
	}

	gmm = EmptyGaussianMixture()
	gmm.NComponents = numComponents
	gmm.Components = make([]*Gaussian, numComponents, numComponents)
	gmm.PosteriorSum = make([]float64, numComponents)
	gmm.LogWeights = make([]float64, numComponents)
	gmm.Diag = true
	gmm.NE = numElements
	gmm.ModelName = name
	gmm.IsTrainable = trainable

	for i, _ := range gmm.Components {
		cname := getComponentName(gmm.ModelName, i, gmm.NComponents)
		gmm.Components[i], e = NewGaussian(gmm.NE, nil, nil, gmm.IsTrainable, gmm.Diag, cname)
		if e != nil {
			return
		}
	}

	// Initialize weights.
	logw := -math.Log(float64(gmm.NComponents))
	floatx.Apply(setValueFunc(logw), gmm.LogWeights, nil)

	e = gmm.Initialize()
	if e != nil {
		return
	}
	return
}

func (gmm *GMM) Initialize() error {

	gmm.tmpProbs = make([]float64, gmm.NComponents)

	// Initialize weights.
	gmm.Weights = make([]float64, gmm.NComponents)
	floatx.Apply(exp, gmm.LogWeights, gmm.Weights)
	return nil
}

// Returns an empty model with the base modeled initialized.
// Use it reading model from Reader.
func EmptyGaussianMixture() *GMM {

	gmm := &GMM{}
	gmm.BaseModel = model.NewBaseModel(model.Modeler(gmm))
	return gmm
}

// Computes log prob for mixture.// SIDE EFFECT => returns logProb of
// Gaussian comp + logWeight in matrix pointed by func arg probs.
func (gmm *GMM) logProbInternal(obs, probs []float64) float64 {

	var max float64 = -math.MaxFloat64

	/* Compute log probabilities for this observation. */
	for i, c := range gmm.Components {
		v1 := c.LogProb(obs)
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

// Returns the log probability.
func (gmm *GMM) LogProb(observation interface{}) float64 {

	obs := observation.([]float64)
	return gmm.logProbInternal(obs, nil)
}

// Returns the probability.
func (gmm *GMM) Prob(observation interface{}) float64 {

	obs := observation.([]float64)
	return math.Exp(gmm.LogProb(obs))
}

/*
  The posterior prob for each mixture component. We approximate the sum using max.

                 p(o|c(i)) p(c(i))
   p(c(i)|o) ~ ---------------------
                max{p(o|c(i)) p(c(i))}

*/

func (gmm *GMM) Update(obs []float64, w float64) error {

	if !gmm.IsTrainable {
		return fmt.Errorf("Attempted to train model [%s] which is not trainable.", gmm.ModelName)
	}

	maxProb := gmm.logProbInternal(obs, gmm.tmpProbs)
	gmm.Likelihood += maxProb
	floatx.Apply(addScalarFunc(-maxProb+math.Log(w)), gmm.tmpProbs, nil)

	// Compute posterior probabilities.
	floatx.Apply(exp, gmm.tmpProbs, nil)

	// Update posterior sum, needed to compute mixture weights.
	floats.Add(gmm.PosteriorSum, gmm.tmpProbs)

	// Update Gaussian components.
	for i, c := range gmm.Components {
		c.Update(obs, gmm.tmpProbs[i])
	}

	// Count number of observations.
	gmm.NSamples += w

	return nil
}

func (gmm *GMM) Estimate() error {

	if !gmm.IsTrainable {
		return fmt.Errorf("Attempted to train model [%s] which is not trainable.", gmm.ModelName)
	}

	// Estimate mixture weights.
	floatx.Apply(scaleFunc(1.0/gmm.NSamples), gmm.PosteriorSum, gmm.Weights)
	floatx.Apply(log, gmm.Weights, gmm.LogWeights)

	// Estimate component density.
	for _, c := range gmm.Components {
		c.Estimate()
	}
	gmm.Iteration += 1

	return nil
}

func (gmm *GMM) Clear() error {

	if !gmm.IsTrainable {
		return fmt.Errorf("Attempted to train model [%s] which is not trainable.", gmm.ModelName)
	}

	for _, c := range gmm.Components {
		c.Clear()
	}
	floatx.Apply(setValueFunc(0), gmm.PosteriorSum, nil)
	gmm.NSamples = 0
	gmm.Likelihood = 0

	return nil
}

// Returns a random GMM vector
func (gmm *GMM) Random(r *rand.Rand) (interface{}, []int, error) {
	// Choose a component using weights
	comp, err := model.RandIntFromDist(gmm.Weights, r)
	if err != nil {
		return nil, nil, err
	}
	// Get a random vector from that component
	return gmm.Components[comp].Random(r)
}

// Returns a random vector using the mean and sd vectors.
func RandomVector(mean, sd []float64, r *rand.Rand) (vec []float64, e error) {

	nrows := len(mean)
	if !floats.EqualLengths(mean, sd) {
		panic(floatx.ErrLength)
	}

	vec = make([]float64, nrows)
	for i := 0; i < nrows; i++ {
		v := r.NormFloat64()*sd[i] + mean[i]
		vec[i] = v
	}
	return
}

// Generates a random Gaussian mixture model using mean and variance vectors as seed.
// Use this function to initialize the GMM before training. The mean and sd
// vector can be estimated from the data set using a Gaussian model.
func RandomGMM(mean, sd []float64, numComponents int,
	name string, seed int64) (gmm *GMM, e error) {

	nrows := len(mean)
	if !floats.EqualLengths(mean, sd) {
		panic(floatx.ErrLength)
	}

	gmm, e = NewGaussianMixture(nrows, numComponents, true, true, name)
	if e != nil {
		return
	}

	r := rand.New(rand.NewSource(seed))
	for _, c := range gmm.Components {
		var rv []float64
		if rv, e = RandomVector(mean, sd, r); e != nil {
			return
		}
		c.Mean = rv
		variance := make([]float64, len(sd))
		floatx.Apply(sq, sd, variance)
		c.setVariance(variance)
	}
	return
}

// Returns the Gaussian components.
/*
func (gmm *GMM) Components() []*Gaussian {
	return gmm.Components
}
*/
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

func (gmm *GMM) Name() string        { return gmm.ModelName }
func (gmm *GMM) NumSamples() float64 { return gmm.NSamples }
func (gmm *GMM) NumElements() int    { return gmm.NE }
func (gmm *GMM) Trainable() bool     { return gmm.IsTrainable }
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
