package model

import (
	"code.google.com/p/biogo.matrix"
	"fmt"
	"math/rand"
	"testing"
)

const epsilon = 0.001

func cmpf64(f1, f2 float64) bool {
	err := f2 - f1
	if err < 0 {
		err = -err
	}
	if err < epsilon {
		return true
	}
	return false
}

// Tests

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

func getRandomVector(mean, std []float64) (*matrix.Dense, error) {

	rand.NewSource(99)
	if len(mean) != len(std) {
		return nil, fmt.Errorf("Cannot generate random vectors length of mean [%d] and std [%d] don't match.",
			len(mean), len(std))
	}
	vector := matrix.MustDense(matrix.ZeroDense(len(mean), 1))
	//r, c := vector.Dims()
	//fmt.Printf("vector r: %d c: %d\n", r, c)

	for i, _ := range mean {
		v := rand.NormFloat64()*std[i] + mean[i]
		//fmt.Printf("len: %d XXX %d: %f\n", len(mean), i, v)
		vector.Set(i, 0, v)
	}

	return vector, nil
}

func TestTrainGaussian(t *testing.T) {

	dim := 8
	mean := []float64{0.1, 0.2, 0.3, 0.4, 1, 1, 1, 1}
	std := []float64{0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4}
	g, e := NewGaussian(dim, nil, nil, true, true, "test training")
	if e != nil {
		t.Fatal(e)
	}
	for i := 0; i < 2000000; i++ {
		rv, err := getRandomVector(mean, std)
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
