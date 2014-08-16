package gaussian

import (
	"math/rand"
	"os"
	"testing"

	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model"
)

const epsilon = 0.004

// Tests

func TestGaussian(t *testing.T) {

	g, e := NewGaussian(10, nil, nil, true, true, "testing")
	if e != nil {
		t.Fatal(e)
	}

	mean := []float64{0.5, 1, 2}
	sd := []float64{1, 1, 1}

	g, e = NewGaussian(3, mean, sd, true, true, "testing")
	if e != nil {
		t.Fatal(e)
	}
	obs := []float64{1, 1, 1}

	p := g.LogProb(obs)
	t.Logf("LogProb: %f", p)
	t.Logf("Prob: %f", g.Prob(obs))

	expected := -3.3818
	if !gjoa.Comparef64(expected, p, epsilon) {
		t.Errorf("Wrong LogProb. Expected: [%f], Got: [%f]", expected, p)
	}
}

func TestWriteReadGaussian(t *testing.T) {

	g, e := NewGaussian(10, nil, nil, true, true, "testing")
	if e != nil {
		t.Fatal(e)
	}

	mean := []float64{0.5, 1, 2}
	//	variance := []float64{1, 2, 4}
	sd := []float64{1, 2, 4}

	g, e = NewGaussian(3, mean, sd, true, true, "testing")
	if e != nil {
		t.Fatal(e)
	}

	fn := os.TempDir() + "gaussian.json"
	g.WriteFile(fn)

	// Create another Gaussian model.
	g0 := EmptyGaussian()
	m1, e1 := g0.ReadFile(fn)
	if e1 != nil {
		t.Fatal(e1)
	}
	g1 := m1.(*Gaussian)

	// Read values from file.
	t.Logf("Original model:\n%+v\n", g)
	t.Logf("New model read from file:\n%+v\n", g1)

	CompareGaussians(t, g, g1, epsilon)
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
	t.Logf("Mean: \n%+v", g.Mean)
	t.Logf("STD: \n%+v", g.StdDev)

	for i, _ := range mean {
		if !gjoa.Comparef64(mean[i], g.Mean[i], epsilon) {
			t.Errorf("Wrong Mean[%d]. Expected: [%f], Got: [%f]",
				i, mean[i], g.Mean[i])
		}
		if !gjoa.Comparef64(std[i], g.StdDev[i], epsilon) {
			t.Errorf("Wrong STD[%d]. Expected: [%f], Got: [%f]",
				i, std[i], g.StdDev[i])
		}
	}
}

func TestCloneGaussian(t *testing.T) {

	dim := 8
	mean := []float64{0.1, 0.2, 0.3, 0.4, 1, 1, 1, 1}
	std := []float64{0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4}
	g, e := NewGaussian(dim, nil, nil, true, true, "test cloning")
	if e != nil {
		t.Fatal(e)
	}
	r := rand.New(rand.NewSource(33))
	for i := 0; i < 2000; i++ {
		rv, err := model.RandNormalVector(mean, std, r)
		if err != nil {
			t.Fatal(err)
		}
		g.Update(rv, 1.0)
	}
	g.Estimate()

	ng := g.Clone()
	if e != nil {
		t.Fatal(e)
	}

	// compare g vs. ng
	type table struct {
		v1, v2 []float64
		name   string
	}

	tab := []table{
		table{g.Sumx, ng.Sumx, "Sumx"},
		table{g.Sumxsq, ng.Sumxsq, "Sumxsq"},
		table{g.Mean, ng.Mean, "Mean"},
		table{g.StdDev, ng.StdDev, "StdDev"},
		table{g.variance, ng.variance, "variance"},
		table{g.varianceInv, ng.varianceInv, "varianceInv"},
		table{[]float64{g.const1}, []float64{ng.const1}, "const1"},
		table{[]float64{g.const2}, []float64{ng.const2}, "const2"},
	}

	// compare slices
	for _, v := range tab {
		gjoa.CompareSliceFloat(t, v.v1, v.v2, "no match: "+v.name, 0.00001)
	}

	if ng.BaseModel.Model == g.BaseModel.Model {
		t.Fatalf("Modeler is the same.")
	}
	if ng.NSamples != g.NSamples {
		t.Fatalf("NSamples doesn't match.")
	}

}
