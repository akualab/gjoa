package gaussian

import (
	"github.com/akualab/gjoa/model"
	"math/rand"
	"testing"
)

const epsilon = 0.004

// Tests

func TestGaussian(t *testing.T) {

	g, e := NewGaussian(10, nil, nil, true, true, "testing")
	if e != nil {
		t.Fatal(e)
	}

	mean := []float64{0.5, 1, 2}
	variance := []float64{1, 1, 1}

	g, e = NewGaussian(3, mean, variance, true, true, "testing")
	if e != nil {
		t.Fatal(e)
	}
	obs := []float64{1, 1, 1}

	p := g.LogProb(obs)
	t.Logf("LogProb: %f", p)
	t.Logf("Prob: %f", g.Prob(obs))

	expected := -3.3818
	if !model.Comparef64(expected, p, epsilon) {
		t.Errorf("Wrong LogProb. Expected: [%f], Got: [%f]", expected, p)
	}
}

func TestTrainGaussian(t *testing.T) {

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	dim := 8
	mean := []float64{0.1, 0.2, 0.3, 0.4, 1, 1, 1, 1}
	std := []float64{0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4}
	g, e := NewGaussian(dim, nil, nil, true, true, "test training")
	if e != nil {
		t.Fatal(e)
	}
	r := rand.New(rand.NewSource(33))
	for i := 0; i < 2000000; i++ {
		rv, err := model.RandNormalVector(mean, std, r)
		if err != nil {
			t.Fatal(err)
		}
		g.Update(rv, 1.0)
	}
	g.Estimate()
	t.Logf("Mean: \n%+v", g.Mean())
	t.Logf("STD: \n%+v", g.StandardDeviation())

	for i, _ := range mean {
		if !model.Comparef64(mean[i], g.Mean()[i], epsilon) {
			t.Errorf("Wrong Mean[%d]. Expected: [%f], Got: [%f]",
				i, mean[i], g.Mean()[i])
		}
		if !model.Comparef64(std[i], g.StandardDeviation()[i], epsilon) {
			t.Errorf("Wrong STD[%d]. Expected: [%f], Got: [%f]",
				i, std[i], g.StandardDeviation()[i])
		}
	}
}
