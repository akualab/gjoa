package gaussian

import (
	"github.com/akualab/gjoa/model"
	"math/rand"
	"testing"
)

func TestGMMName(t *testing.T) {

	gmm, e := NewGaussianMixture(4, 123, true, true, "mygmm")
	if e != nil {
		t.Fatal(e)
	}

	// for i, c := range gmm.Components() {
	// 	t.Logf("Name for comp #%4d: %s", i, c.Name())
	// }

	name := gmm.Components()[111].Name()
	if name != "mygmm-111" {
		t.Errorf("Wrong component name in gmm. Expected: [mygmm-111], Got: [%s]", name)
	}
}

// Trains a GMM as follows:
// 1 - Estimate a Gaussian model params for the trainign set.
// 2 - Use teh mean and variance of the training set to generate
//     a random GMM to be used as seed.
// 3 - Run several iterations of the GMM max likelihood training algorithm
//     to estimate the GMM weights and the Gaussian component mean, and
//     variance vectors.
func TestTrainGMM(t *testing.T) {

	var seed int64 = 33
	numComp := 2
	numIter := 10
	numObs := 1000000

	mean0 := []float64{1, 2}
	std0 := []float64{0.3, 0.3}
	mean1 := []float64{4, 4}
	std1 := []float64{1, 1}
	dim := len(mean0)

	gmm, e := NewGaussianMixture(dim, numComp, true, true, "mygmm")
	if e != nil {
		t.Fatal(e)
	}
	t.Logf("Initial Weights: \n%+v", gmm.weights)

	// Estimate mean variance of the data.
	g, e := NewGaussian(dim, nil, nil, true, true, "test training")
	if e != nil {
		t.Fatal(e)
	}
	r := rand.New(rand.NewSource(seed))
	for i := 0; i < numObs; i++ {
		rv, err := model.GetRandomVector(mean0, std0, r)
		if err != nil {
			t.Fatal(err)
		}
		g.Update(rv, 1.0)
		rv, err = model.GetRandomVector(mean1, std1, r)
		if err != nil {
			t.Fatal(err)
		}
		g.Update(rv, 1.0)
	}
	g.Estimate()
	t.Logf("Gaussian Model for training set:")
	t.Logf("Mean: \n%+v", g.Mean())
	t.Logf("STD: \n%+v", g.StandardDeviation())

	// Use the estimated mean and variance to generate a seed GMM.
	gmm, e = RandomGMM(g.Mean(), g.Variance(), numComp,
		"mygmm", 99)
	if e != nil {
		t.Fatal(e)
	}

	for iter := 0; iter < numIter; iter++ {
		t.Logf("Starting GMM trainign iteration %d.", iter)

		// Reset the same random number generator to make sure we use the
		// same observations in each iterations.
		r := rand.New(rand.NewSource(seed))

		// Update GMM stats.
		for i := 0; i < numObs; i++ {
			rv, err := model.GetRandomVector(mean0, std0, r)
			if err != nil {
				t.Fatal(err)
			}
			gmm.Update(rv, 1.0)
			rv, err = model.GetRandomVector(mean1, std1, r)
			if err != nil {
				t.Fatal(err)
			}
			gmm.Update(rv, 1.0)
		}

		// Estimates GMM params.
		gmm.Estimate()

		t.Logf("Iter: %d", iter)
		t.Logf("GMM: %+v", gmm)
		t.Logf("Weights: \n%+v", gmm.weights)
		t.Logf("Likelihood: %f", gmm.totalLikelihood)
		t.Logf("Num Samples: %f", gmm.numSamples)
		for _, c := range gmm.components {
			t.Logf("%s: Mean: \n%+v", c.Name(), c.Mean())
			t.Logf("%s: STD: \n%+v", c.Name(), c.StandardDeviation())
		}

		// Prepare for next iteration.
		gmm.Clear()
	}

	for i := 0; i < dim; i++ {
		g := gmm.components[1]
		if !model.Comparef64(mean0[i], g.Mean()[i]) {
			t.Errorf("Wrong Mean[%d]. Expected: [%f], Got: [%f]",
				i, mean0[i], g.Mean()[i])
		}
		if !model.Comparef64(std0[i], g.StandardDeviation()[i]) {
			t.Errorf("Wrong STD[%d]. Expected: [%f], Got: [%f]",
				i, std0[i], g.StandardDeviation()[i])
		}
	}

	for i := 0; i < dim; i++ {
		g := gmm.components[0]
		if !model.Comparef64(mean1[i], g.Mean()[i]) {
			t.Errorf("Wrong Mean[%d]. Expected: [%f], Got: [%f]",
				i, mean1[i], g.Mean()[i])
		}
		if !model.Comparef64(std1[i], g.StandardDeviation()[i]) {
			t.Errorf("Wrong STD[%d]. Expected: [%f], Got: [%f]",
				i, std1[i], g.StandardDeviation()[i])
		}
	}

	if !model.Comparef64(0.5, gmm.weights[0]) {
		t.Errorf("Wrong weights[0]. Expected: [%f], Got: [%f]",
			0.5, gmm.weights[0])
	}

	if !model.Comparef64(0.5, gmm.weights[1]) {
		t.Errorf("Wrong weights[0]. Expected: [%f], Got: [%f]",
			0.5, gmm.weights[1])
	}

}
