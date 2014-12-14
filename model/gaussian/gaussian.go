// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gaussian

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/akualab/gjoa/floatx"
	"github.com/akualab/gjoa/model"
	"github.com/golang/glog"
	"github.com/gonum/floats"
)

const (
	smallSD       = 0.01
	smallVar      = smallSD * smallSD
	minNumSamples = 0.01
	defaultSeed   = 33
)

// Model is a multivariate Gaussian distribution.
type Model struct {
	ModelName   string    `json:"name,omitempty"`
	ModelDim    int       `json:"dim"`
	NSamples    float64   `json:"nsamples"`
	Diag        bool      `json:"diag"`
	Sumx        []float64 `json:"sumx,omitempty"`
	Sumxsq      []float64 `json:"sumx_sq,omitempty"`
	Mean        []float64 `json:"mean"`
	StdDev      []float64 `json:"sd"`
	Seed        int64     `json:"seed"`
	variance    []float64
	varianceInv []float64
	tmpArray    []float64
	const1      float64 // -(N/2)log(2PI) Depends only on ModelDim.
	const2      float64 // const1 - sum(log sigma_i) Also depends on variance.
	rand        *rand.Rand
}

// Option type is used to pass options to NewModel().
type Option func(*Model)

// NewModel creates a new Gaussian model.
func NewModel(dim int, options ...Option) *Model {

	g := &Model{
		ModelName:   "Gaussian",
		ModelDim:    dim,
		Diag:        true,
		variance:    make([]float64, dim),
		varianceInv: make([]float64, dim),
		tmpArray:    make([]float64, dim),
		Seed:        model.DefaultSeed,
	}

	// Set options.
	for _, option := range options {
		option(g)
	}
	g.rand = rand.New(rand.NewSource(g.Seed))
	if len(g.Sumx) == 0 {
		g.Sumx = make([]float64, dim)
	}
	if len(g.Sumxsq) == 0 {
		g.Sumxsq = make([]float64, dim)
	}
	if g.Mean == nil {
		g.Mean = make([]float64, dim)
	}
	if g.StdDev == nil {
		g.StdDev = make([]float64, dim)
		floatx.Apply(floatx.SetValueFunc(smallSD), g.StdDev, nil)
	}

	floatx.Sq(g.variance, g.StdDev)

	// Initializes variance, varianceInv, and StdDev.
	g.setVariance(g.variance)

	floatx.Log(g.tmpArray, g.variance)
	g.const1 = -float64(g.ModelDim) * math.Log(2.0*math.Pi) / 2.0
	g.const2 = g.const1 - floats.Sum(g.tmpArray)/2.0
	return g
}

// Update updates sufficient statistics using observations.
func (g *Model) Update(x model.Observer, w func(model.Obs) float64) error {
	c, e := x.ObsChan()
	if e != nil {
		return e
	}
	for v := range c {
		g.UpdateOne(v, w(v))
	}
	return nil
}

// Predict returns a hypothesis given the observation.
func (g *Model) Predict(x model.Observer) ([]model.Labeler, error) {

	glog.Fatal("Predict method not implemented.")
	return nil, nil
}

// Sample returns a Gaussian sample.
func (g *Model) Sample() model.Obs {
	obs, e := model.RandNormalVector(g.Mean, g.StdDev, g.rand)
	if e != nil {
		glog.Fatal(e)
	}
	return model.NewFloatObs(obs, model.SimpleLabel{})
}

// SampleChan returns a channel with "size" samples drawn from teh model.
// The sequence ends when the channel closes.
func (g *Model) SampleChan(size int) <-chan model.Obs {

	if len(g.Mean) == 0 {
		glog.Fatal("Parameter Mean is missing.")
	}
	if len(g.StdDev) == 0 {
		glog.Fatal("Parameter StdDev is missing.")
	}
	if g.rand == nil {
		glog.Fatal("Random value generator is missing.")
	}
	c := make(chan model.Obs, 1000)
	go func() {
		for i := 0; i < size; i++ {
			c <- g.Sample()
		}
		close(c)
	}()
	return c
}

// LogProb returns log probability for observation.
func (g *Model) LogProb(obs model.Obs) float64 {

	return g.logProb(obs.Value().([]float64))
}

func (g *Model) logProb(obs []float64) (v float64) {

	for i, x := range obs {
		s := g.Mean[i] - x
		v += s * s * g.varianceInv[i] / 2.0
	}
	v = g.const2 - v
	return
}

func (g *Model) prob(obs []float64) float64 {

	return math.Exp(g.logProb(obs))
}

// UpdateOne updates sufficient statistics using one observation.
func (g *Model) UpdateOne(o model.Obs, w float64) {

	/* Update sufficient statistics. */
	obs, _ := model.ObsToF64(o)
	floatx.Apply(floatx.ScaleFunc(w), obs, g.tmpArray)
	floats.Add(g.Sumx, g.tmpArray)
	floatx.Sq(g.tmpArray, obs)
	floats.Scale(w, g.tmpArray)
	floats.Add(g.Sumxsq, g.tmpArray)
	g.NSamples += w
}

// Estimate computes model parameters using sufficient statistics.
func (g *Model) Estimate() error {

	if g.NSamples > minNumSamples {

		/* Estimate the mean. */
		floatx.Apply(floatx.ScaleFunc(1.0/g.NSamples), g.Sumx, g.Mean)
		/*
		 * Estimate the variance. sigma_sq = 1/n (sumxsq - 1/n sumx^2) or
		 * 1/n sumxsq - mean^2.
		 */
		tmp := g.variance // borrow as an intermediate array.

		//		floatx.Apply(sq, g.Mean, g.tmpArray)
		floatx.Sq(g.tmpArray, g.Mean)
		floatx.Apply(floatx.ScaleFunc(1.0/g.NSamples), g.Sumxsq, tmp)
		floats.SubTo(g.variance, tmp, g.tmpArray)
		floatx.Apply(floatx.Floorv(smallVar), g.variance, nil)
	} else {

		/* Not enough training sample. */
		floatx.Apply(floatx.SetValueFunc(smallVar), g.variance, nil)
		floatx.Apply(floatx.SetValueFunc(0), g.Mean, nil)
	}
	g.setVariance(g.variance) // to update varInv and stddev.

	/* Update log Gaussian constant. */
	floatx.Log(g.tmpArray, g.variance)
	g.const2 = g.const1 - floats.Sum(g.tmpArray)/2.0

	return nil
}

// Clear resets sufficient statistics.
func (g *Model) Clear() {

	floatx.Apply(floatx.SetValueFunc(0), g.Sumx, nil)
	floatx.Apply(floatx.SetValueFunc(0), g.Sumxsq, nil)
	g.NSamples = 0
}

func (g *Model) setVariance(variance []float64) {
	copy(g.variance, variance)
	floatx.Apply(floatx.Inv, g.variance, g.varianceInv)
	g.StdDev = g.standardDeviation()
}

func (g *Model) standardDeviation() (sd []float64) {

	sd = make([]float64, g.ModelDim)
	floatx.Sqrt(sd, g.variance)
	return
}

// Dim is the dimensionality of the observation vector.
func (g *Model) Dim() int { return g.ModelDim }

// Name returns the name of the model.
func (g *Model) Name() string {
	return g.ModelName
}

// Options

// Mean is an option function. Use it to set a
// mean vector when creating a new Gaussian model.
func Mean(mean []float64) Option {
	return func(m *Model) {
		m.Mean = mean
	}
}

// StdDev is an option function. Use it to set a
// standard deviation vector when creating a
// new Gaussian model.
func StdDev(sd []float64) Option {
	return func(g *Model) { g.StdDev = sd }
}

// SetName sets a name for the model.
func (g *Model) setName(name string) {
	g.ModelName = name
}

// Name is an option to set the model name.
func Name(name string) Option {
	return func(g *Model) { g.setName(name) }
}

// Seed sets a seed value for random functions.
func Seed(seed int64) Option {
	return func(g *Model) { g.Seed = seed }
}

// Clone create a clone of src.
func Clone(src *Model) Option {
	g := src
	return func(ng *Model) {
		ng.ModelName = g.ModelName
		ng.ModelDim = g.ModelDim
		ng.NSamples = g.NSamples
		ng.Diag = g.Diag

		ng.Sumx = g.Sumx
		ng.Sumxsq = g.Sumxsq

		ng.Mean = g.Mean
		ng.StdDev = g.StdDev
	}
}

// IO

// Read unmarshals json data from an io.Reader into a model struct.
func Read(r io.Reader) (*Model, error) {

	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}

	// Get a Model object.
	m := &Model{}
	e := json.Unmarshal(b, m)

	if e != nil {
		return nil, e
	}
	m = NewModel(m.ModelDim, Clone(m))
	return m, nil
}

// ReadFile unmarshals json data from a file into a model struct.
func ReadFile(fn string) (*Model, error) {

	f, err := os.Open(fn)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	glog.Infof("Reading model from file %s.", fn)
	return Read(f)
}

// Write writes the model to an io.Writer.
func (g *Model) Write(w io.Writer) error {

	b, err := json.Marshal(g)
	if err != nil {
		return err
	}
	_, e := w.Write(b)
	return e
}

// WriteFile writes the model to file.
func (g *Model) WriteFile(fn string) error {

	e := os.MkdirAll(filepath.Dir(fn), 0755)
	if e != nil {
		return e
	}
	f, err := os.Create(fn)
	if err != nil {
		return err
	}
	defer f.Close()

	ee := g.Write(f)
	if ee != nil {
		return ee
	}

	glog.Infof("Wrote model \"%s\" to file %s.", g.Name(), fn)
	return nil
}
