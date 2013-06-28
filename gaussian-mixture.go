package gjøa

import (
	"code.google.com/p/biogo.matrix"
	"fmt"
	"math"
	"math/rand"
)

type GMMConfig struct {
}

type GMM struct {
	model
	diagonal        bool
	trainingMethod  int
	numComponents   int
	posteriorSum    *matrix.Dense
	weights         *matrix.Dense
	logWeights      *matrix.Dense
	tmpProbs1       *matrix.Dense
	tmpProbs2       *matrix.Dense
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
			model: model{
				numElements: numElements,
				name:        name,
				trainable:   trainable,
			},
		}, nil
	}

	gmm = &GMM{
		numComponents: numComponents,
		components:    make([]*Gaussian, numComponents, numComponents),
		posteriorSum:  matrix.MustDense(matrix.ZeroDense(numComponents, 1)),
		weights:       matrix.MustDense(matrix.ZeroDense(numComponents, 1)),
		logWeights:    matrix.MustDense(matrix.ZeroDense(numComponents, 1)),
		tmpProbs1:     matrix.MustDense(matrix.ZeroDense(numComponents, 1)),
		tmpProbs2:     matrix.MustDense(matrix.ZeroDense(numComponents, 1)),
		diagonal:      true,
		model: model{
			numElements: numElements,
			name:        name,
			trainable:   trainable,
		},
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
	gmm.weights.ApplyDense(setValueFunc(w), gmm.weights)
	gmm.logWeights.ApplyDense(setValueFunc(math.Log(w)), gmm.logWeights)

	return
}

// Computes log prob for mixture.// SIDE EFFECT => returns logProb of
// Gaussian comp + logWeight in matrix pointed by func arg probs.
func (gmm *GMM) logProbInternal(obs, probs *matrix.Dense) float64 {

	//fmt.Printf("obs: \n%+v\n", obs)

	var max float64 = -math.MaxFloat64

	/* Compute log probabilities for this observation. */
	for i, c := range gmm.components {
		//v := c.LogProb(obs) + gmm.logWeights.At(i, 0)
		v1 := c.LogProb(obs)
		v2 := gmm.logWeights.At(i, 0)
		v := v1 + v2
		//fmt.Printf("comp %2d: logProbInternal: logProb: %f, logW: %f\n", i, v1, v2)

		if probs != nil {
			probs.Set(i, 0, v)
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
func (gmm *GMM) LogProb(obs *matrix.Dense) float64 {

	return gmm.logProbInternal(obs, nil)
}

// Returns the probability.
func (gmm *GMM) Prob(obs *matrix.Dense) float64 {
	return math.Exp(gmm.LogProb(obs))
}

// Update model statistics.
func (gmm *GMM) Update(obs *matrix.Dense) error {

	if !gmm.trainable {
		return fmt.Errorf("Attempted to train model [%s] which is not trainable.", gmm.Name())
	}

	maxProb := gmm.logProbInternal(obs, gmm.tmpProbs2)
	gmm.totalLikelihood += maxProb
	gmm.tmpProbs2.ApplyDense(addScalarFunc(-maxProb), gmm.tmpProbs2)

	// Compute posterior probabilities.
	gmm.tmpProbs2.ApplyDense(exp, gmm.tmpProbs2)

	// Update posterior sum, needed to compute mixture weights.
	gmm.posteriorSum.Add(gmm.tmpProbs2, gmm.posteriorSum)

	// Update Gaussian components.
	for i, c := range gmm.components {
		c.WUpdate(obs, gmm.tmpProbs2.At(i, 0))
	}

	// Count number of observations.
	gmm.numSamples += 1

	return nil
}

func (gmm *GMM) Estimate() error {

	if !gmm.trainable {
		return fmt.Errorf("Attempted to train model [%s] which is not trainable.", gmm.Name())
	}

	// Estimate mixture weights.
	gmm.posteriorSum.Scalar(1.0/gmm.numSamples, gmm.weights)
	gmm.weights.ApplyDense(log, gmm.logWeights)

	// Estimate component density.
	for _, c := range gmm.components {
		c.Estimate()
	}
	gmm.iteration += 1

	return nil
}

func (gmm *GMM) Clear() error {

	if !gmm.trainable {
		return fmt.Errorf("Attempted to train model [%s] which is not trainable.", gmm.Name())
	}

	for _, c := range gmm.components {
		c.Clear()
	}
	gmm.posteriorSum.ApplyDense(setValueFunc(0), gmm.posteriorSum)
	gmm.numSamples = 0
	gmm.totalLikelihood = 0

	return nil
}

// Returns a random vector using the mean and variance vector.
func RandomVector(mean, variance *matrix.Dense, r *rand.Rand) (vec *matrix.Dense, e error) {

	var nrows int
	if nrows, e = CheckVectorShape(mean, variance); e != nil {
		return
	}

	vec = matrix.MustDense(matrix.ZeroDense(nrows, 1))
	for i := 0; i < nrows; i++ {
		std := math.Sqrt(variance.At(i, 0))
		v := r.NormFloat64()*std + mean.At(i, 0)
		vec.Set(i, 0, v)
	}
	return
}

// Generates a random Gaussian mixture model using mean and variance vectors as seed.
// Use this function to initialize the GMM before training. The mean and variance
// vector can be estimated from the data set using a Gaussian model.
func RandomGMM(mean, variance *matrix.Dense, numComponents int,
	name string, seed int64) (gmm *GMM, e error) {

	var nrows int
	if nrows, e = CheckVectorShape(mean, variance); e != nil {
		return
	}

	gmm, e = NewGaussianMixture(nrows, numComponents, true, true, name)
	if e != nil {
		return
	}

	r := rand.New(rand.NewSource(seed))
	for _, c := range gmm.components {
		var rv *matrix.Dense
		if rv, e = RandomVector(mean, variance, r); e != nil {
			return
		}
		c.SetMean(rv)
		varianceCopy := matrix.MustDense(matrix.ZeroDense(nrows, 1))
		variance.Clone(varianceCopy)
		c.SetVariance(varianceCopy)
	}
	return
}

// Returns the Gaussian components.
func (gmm *GMM) Components() []*Gaussian {
	return gmm.components
}

// Checks that num cols is one and that num rows match.
func CheckVectorShape(v1, v2 *matrix.Dense) (int, error) {
	r1, c1 := v1.Dims()
	r2, c2 := v2.Dims()
	if c1 != 1 || c2 != 1 {
		return 0, fmt.Errorf("Num cols must be one. v1: [%d,%d], v2: [%d,%d]",
			r1, c1, r2, c2)
	}
	if r1 != r2 || c1 != c2 {
		return 0, fmt.Errorf("Shape of v1 and v2 matrices must match. v1: [%d,%d], v2: [%d,%d]",
			r1, c1, r2, c2)
	}
	return r1, nil
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
