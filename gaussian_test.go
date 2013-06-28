package gjøa

import (
	"code.google.com/p/biogo.matrix"
	"math/rand"
	"testing"
)

// Tests

func TestGaussian(t *testing.T) {

	g, e := NewGaussian(10, nil, nil, true, true, "testing")
	if e != nil {
		t.Fatal(e)
	}
	//t.Logf("Gaussian: %+v", g)

	mean, e2 := matrix.NewDense([][]float64{{0.5}, {1}, {2}})
	if e2 != nil {
		t.Fatal(e2)
	}
	variance, ev := matrix.NewDense([][]float64{{1}, {1}, {1}})
	if ev != nil {
		t.Fatal(ev)
	}

	g, e = NewGaussian(3, mean, variance, true, true, "testing")
	if e != nil {
		t.Fatal(e)
	}
	obs, e3 := matrix.NewDense([][]float64{{1}, {1}, {1}})
	if e3 != nil {
		t.Fatal(e3)
	}
	p := g.LogProb(obs)
	//t.Logf("Gaussian: %+v", g)
	t.Logf("LogProb: %f", p)
	t.Logf("Prob: %f", g.Prob(obs))
	// -3.3818

	expected := -3.3818
	if !cmpf64(expected, p) {
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
		rv, err := getRandomVector(mean, std, r)
		if err != nil {
			t.Fatal(err)
		}
		g.Update(rv)
	}
	g.Estimate()
	//t.Logf("Gaussian: %+v", g)
	t.Logf("Mean: \n%+v", g.Mean())
	t.Logf("STD: \n%+v", g.StandardDeviation())

	for i, _ := range mean {
		if !cmpf64(mean[i], g.Mean().At(i, 0)) {
			t.Errorf("Wrong Mean[%d]. Expected: [%f], Got: [%f]",
				i, mean[i], g.Mean().At(i, 0))
		}
		if !cmpf64(std[i], g.StandardDeviation().At(i, 0)) {
			t.Errorf("Wrong STD[%d]. Expected: [%f], Got: [%f]",
				i, std[i], g.StandardDeviation().At(i, 0))
		}
	}
}
