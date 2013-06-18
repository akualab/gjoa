package model

import (
	"code.google.com/p/biogo.matrix"
	"fmt"
	"math"
	"math/rand"
)

const (

	// Estimate mean and variance of Gaussian distribution in the first
	// iteration. Create the target number of Gaussian components
	// (numComponents) in the mixture at the end of the first iteration
	// using the estimated mean and variance.
	GMM_STEP = iota

	// Double the number of Gaussian components at the end of each
	// iteration.
	GMM_DOUBLE

	// Do not allocate structures for training.
	GMM_NONE
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
	//fmt.Printf("tmpProbs2: \n%+v\n", gmm.tmpProbs2)

	gmm.totalLikelihood += maxProb
	gmm.tmpProbs2.ApplyDense(addScalarFunc(-maxProb), gmm.tmpProbs2)

	//fmt.Printf("%8.f: maxProb: %f\n", gmm.numSamples, maxProb)
	//fmt.Printf("tmpProbs2: \n%+v\n", gmm.tmpProbs2)

	// Compute posterior probabilities.
	gmm.tmpProbs2.ApplyDense(exp, gmm.tmpProbs2)

	//fmt.Printf("posterior: \n%+v\n", gmm.tmpProbs2)

	// Update posterior sum, needed to compute mixture weights.
	gmm.posteriorSum.Add(gmm.tmpProbs2, gmm.posteriorSum)

	//fmt.Printf("posterior sum: \n%+v\n", gmm.posteriorSum)

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

	// After the first iteration, we can estimate the target number of
	// mixture components.
	// if (iteration == 0 && trainMethod == TrainMethod.STEP) {
	//     increaseNumComponents(numComponents);
	// }
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

/*
 * This method is used in {@link TrainMethod#STEP} Convert to a mixture with
 * N components. This method guarantees that the data structures are created
 * and that all the variables are set for starting a new training iteration.
 */
// private void increaseNumComponents(int newNumComponents) {

//     /*
//      * We use the Gaussian distribution of the parent GMM to create the
//      * children.
//      */

//     /* Get mean and variance from parent before we allocate resized data structures. */
//     D1Matrix64F mean = MatrixOps.doubleArrayToMatrix(this.components[0]
//             .getMean());
//     D1Matrix64F variance = MatrixOps.doubleArrayToMatrix(this.components[0]
//             .getVariance());

//     /* Throw away all previous data structures. */
//     allocateTrainDataStructures(newNumComponents);

//     /*
//      * Create new mixture components. Abandon the old ones. We already got
//      * the mean and variance in the previous step.
//      */
//     for (int i = 0; i < newNumComponents; i++) {
//         components[i].setMean(MatrixOps.createRandom(i, mean, variance));
//         components[i].setVariance(new DenseMatrix64F(variance));
//     }
// }

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
// We can use this function to initialize the GMM before training. The mean an variance
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
