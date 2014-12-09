package gaussian

import (
	"math/rand"
	"testing"

	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model"
)

const epsilon = 0.004

// Tests

func TestGaussian(t *testing.T) {

	g := NewGaussian(GaussianParam{
		NumElements: 10,
		Name:        "testing",
	})

	mean := []float64{0.5, 1, 2}
	sd := []float64{1, 1, 1}

	g = NewGaussian(GaussianParam{
		NumElements: 3,
		Mean:        mean,
		StdDev:      sd,
		Name:        "testing",
	})

	obs := []float64{1, 1, 1}

	p := g.logProb(obs)
	t.Logf("LogProb: %f", p)
	t.Logf("Prob: %f", g.prob(obs))

	expected := -3.3818
	if !gjoa.Comparef64(expected, p, epsilon) {
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
	g := NewGaussian(GaussianParam{
		NumElements: dim,
		Name:        "test training",
	})

	r := rand.New(rand.NewSource(33))
	for i := 0; i < 2000000; i++ {
		rv, err := model.RandNormalVector(mean, std, r)
		if err != nil {
			t.Fatal(err)
		}
		g.UpdateOne(F64ToObs(rv), 1.0)
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

func TestTrainGaussian2(t *testing.T) {

	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}
	dim := 8
	numSamp := 2000000

	// Use a FloatObserver.
	values := make([][]float64, numSamp, numSamp)
	labels := make([]model.SimpleLabel, numSamp, numSamp)
	mean := []float64{0.1, 0.2, 0.3, 0.4, 1, 1, 1, 1}
	std := []float64{0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4}
	g := NewGaussian(GaussianParam{
		NumElements: dim,
		Name:        "test training",
	})

	r := rand.New(rand.NewSource(33))
	for i := 0; i < numSamp; i++ {
		rv, err := model.RandNormalVector(mean, std, r)
		if err != nil {
			t.Fatal(err)
		}
		values[i] = rv
	}

	fo, err := model.NewFloatObserver(values, labels)
	if err != nil {
		t.Fatal(err)
	}

	g.Update(fo, model.NoWeight)
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
	g := NewGaussian(GaussianParam{
		NumElements: dim,
		Name:        "test cloning",
	})

	r := rand.New(rand.NewSource(33))
	for i := 0; i < 2000; i++ {
		rv, err := model.RandNormalVector(mean, std, r)
		if err != nil {
			t.Fatal(err)
		}
		g.UpdateOne(F64ToObs(rv), 1.0)
	}
	g.Estimate()

	ng := g.Clone()

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

	//	if ng.BaseModel.Model == g.BaseModel.Model {
	//		t.Fatalf("Modeler is the same.")
	//	}
	if ng.NSamples != g.NSamples {
		t.Fatalf("NSamples doesn't match.")
	}
}

// Train without using sampler.
func BenchmarkTrain(b *testing.B) {

	dim := 8
	mean := []float64{0.1, 0.2, 0.3, 0.4, 1, 1, 1, 1}
	std := []float64{0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4}
	g := NewGaussian(GaussianParam{
		NumElements: dim,
		Name:        "test training",
	})

	r := rand.New(rand.NewSource(33))
	buf := make([][]float64, 2000000, 2000000)
	for i := 0; i < 2000000; i++ {
		rv, err := model.RandNormalVector(mean, std, r)
		if err != nil {
			b.Fatal(err)
		}
		buf[i] = rv
	}

	for i := 0; i < b.N; i++ {
		for i := 0; i < 2000000; i++ {
			g.UpdateOne(F64ToObs(buf[i]), 1.0)
		}
		g.Estimate()
		g.Clear()
	}
}

// Train using sampler.
func BenchmarkTrain2(b *testing.B) {

	dim := 8
	numSamp := 2000000

	// Use a FloatObserver.
	fs, err := model.NewFloatObserver(make([][]float64, numSamp, numSamp),
		make([]model.SimpleLabel, numSamp, numSamp))
	if err != nil {
		b.Fatal(err)
	}
	mean := []float64{0.1, 0.2, 0.3, 0.4, 1, 1, 1, 1}
	std := []float64{0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4}
	g := NewGaussian(GaussianParam{
		NumElements: dim,
		Name:        "test training",
	})

	r := rand.New(rand.NewSource(33))
	for i := 0; i < numSamp; i++ {
		rv, err := model.RandNormalVector(mean, std, r)
		if err != nil {
			b.Fatal(err)
		}
		fs.Values[i] = rv
	}
	for i := 0; i < b.N; i++ {
		g.Update(fs, model.NoWeight)
		g.Estimate()
		g.Clear()
	}
}
