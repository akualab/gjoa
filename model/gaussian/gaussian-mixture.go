package gaussian

import (
	"fmt"
	"github.com/akualab/gjoa/floatx"
	"github.com/gonum/floats"
	"math"
	"math/rand"
)

type GMMConfig struct {
}

type GMM struct {
	name            string
	numElements     int
	trainable       bool
	numSamples      float64
	diagonal        bool
	trainingMethod  int
	numComponents   int
	posteriorSum    []float64
	weights         []float64
	logWeights      []float64
	tmpProbs1       []float64
	tmpProbs2       []float64
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
			numElements:   numElements,
			name:          name,
			trainable:     trainable,
		}, nil
	}

	gmm = &GMM{
		numComponents: numComponents,
		components:    make([]*Gaussian, numComponents, numComponents),
		posteriorSum:  make([]float64, numComponents),
		weights:       make([]float64, numComponents),
		logWeights:    make([]float64, numComponents),
		tmpProbs1:     make([]float64, numComponents),
		tmpProbs2:     make([]float64, numComponents),
		diagonal:      true,
		numElements:   numElements,
		name:          name,
		trainable:     trainable,
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
	floatx.Apply(setValueFunc(w), gmm.weights, nil)
	floatx.Apply(setValueFunc(math.Log(w)), gmm.logWeights, nil)
	return
}

// Computes log prob for mixture.// SIDE EFFECT => returns logProb of
// Gaussian comp + logWeight in matrix pointed by func arg probs.
func (gmm *GMM) logProbInternal(obs, probs []float64) float64 {

	var max float64 = -math.MaxFloat64

	/* Compute log probabilities for this observation. */
	for i, c := range gmm.components {
		v1 := c.LogProb(obs)
		v2 := gmm.logWeights[i]
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
func (gmm *GMM) LogProb(obs []float64) float64 {

	return gmm.logProbInternal(obs, nil)
}

// Returns the probability.
func (gmm *GMM) Prob(obs []float64) float64 {
	return math.Exp(gmm.LogProb(obs))
}

/*
  The posterior prob for each mixture component. We approximate the sum using max.

                 p(o|c(i)) p(c(i))
   p(c(i)|o) ~ ---------------------
                max{p(o|c(i)) p(c(i))}

   The vector gmm.posteriorSum has te
*/

func (gmm *GMM) Update(obs []float64, w float64) error {

	if !gmm.trainable {
		return fmt.Errorf("Attempted to train model [%s] which is not trainable.", gmm.name)
	}

	maxProb := gmm.logProbInternal(obs, gmm.tmpProbs2)
	gmm.totalLikelihood += maxProb
	floatx.Apply(addScalarFunc(-maxProb+math.Log(w)), gmm.tmpProbs2, nil)

	// Compute posterior probabilities.
	floatx.Apply(exp, gmm.tmpProbs2, nil)

	// Update posterior sum, needed to compute mixture weights.
	floats.Add(gmm.posteriorSum, gmm.tmpProbs2)

	// Update Gaussian components.
	for i, c := range gmm.components {
		c.Update(obs, gmm.tmpProbs2[i])
	}

	// Count number of observations.
	gmm.numSamples += w

	return nil
}

func (gmm *GMM) Estimate() error {

	if !gmm.trainable {
		return fmt.Errorf("Attempted to train model [%s] which is not trainable.", gmm.name)
	}

	// Estimate mixture weights.
	floatx.Apply(scaleFunc(1.0/gmm.numSamples), gmm.posteriorSum, gmm.weights)
	floatx.Apply(log, gmm.weights, gmm.logWeights)

	// Estimate component density.
	for _, c := range gmm.components {
		c.Estimate()
	}
	gmm.iteration += 1

	return nil
}

func (gmm *GMM) Clear() error {

	if !gmm.trainable {
		return fmt.Errorf("Attempted to train model [%s] which is not trainable.", gmm.name)
	}

	for _, c := range gmm.components {
		c.Clear()
	}
	floatx.Apply(setValueFunc(0), gmm.posteriorSum, nil)
	gmm.numSamples = 0
	gmm.totalLikelihood = 0

	return nil
}

// Returns a random vector using the mean and variance vector.
func RandomVector(mean, variance []float64, r *rand.Rand) (vec []float64, e error) {

	nrows := len(mean)
	if !floats.EqualLengths(mean, variance) {
		panic(floatx.ErrLength)
	}

	vec = make([]float64, nrows)
	for i := 0; i < nrows; i++ {
		std := math.Sqrt(variance[i])
		v := r.NormFloat64()*std + mean[i]
		vec[i] = v
	}
	return
}

// Generates a random Gaussian mixture model using mean and variance vectors as seed.
// Use this function to initialize the GMM before training. The mean and variance
// vector can be estimated from the data set using a Gaussian model.
func RandomGMM(mean, variance []float64, numComponents int,
	name string, seed int64) (gmm *GMM, e error) {

	nrows := len(mean)
	if !floats.EqualLengths(mean, variance) {
		panic(floatx.ErrLength)
	}

	gmm, e = NewGaussianMixture(nrows, numComponents, true, true, name)
	if e != nil {
		return
	}

	r := rand.New(rand.NewSource(seed))
	for _, c := range gmm.components {
		var rv []float64
		if rv, e = RandomVector(mean, variance, r); e != nil {
			return
		}
		c.SetMean(rv)
		c.SetVariance(variance)
	}
	return
}

// Returns the Gaussian components.
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

func (gmm *GMM) Name() string        { return gmm.name }
func (gmm *GMM) NumSamples() float64 { return gmm.numSamples }
func (gmm *GMM) NumElements() int    { return gmm.numElements }
func (gmm *GMM) Trainable() bool     { return gmm.trainable }
