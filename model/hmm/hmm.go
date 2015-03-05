// Copyright (c) 2015 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

import (
	"fmt"
	"math"
	"reflect"
	"strings"

	"github.com/akualab/gjoa/model"
	"github.com/akualab/narray"
	"github.com/golang/glog"
)

// HMM network implementation.

// Set is a collection of unique HMMs.
type Set struct {
	Nets   []*Net
	byName map[string]*Net
}

// NewSet creates a new set of hmms.
func NewSet(nets ...*Net) (*Set, error) {
	ms := &Set{
		Nets:   make([]*Net, 0),
		byName: make(map[string]*Net),
	}
	for _, v := range nets {
		err := ms.add(v)
		if err != nil {
			return nil, err
		}
	}
	return ms, nil
}

// add models to set
// to keep things consistent, always use this method to add a net to set
func (ms *Set) add(m *Net) error {

	// check that name is unique
	_, ok := ms.byName[m.Name]
	if ok {
		return fmt.Errorf("tried to add net with duplicate name [%s]", m.Name)
	}
	m.id = len(ms.Nets) // asigns unique id
	ms.Nets = append(ms.Nets, m)
	ms.byName[m.Name] = m
	return nil
}

func (ms *Set) net(name string) (*Net, bool) {

	// check that model for name exist
	m, ok := ms.byName[name]
	if !ok {
		return nil, false
	}
	return m, true
}

// Net is an hmm network with a single non-emmiting entry state and
// a single non-emmiting exit state.
type Net struct {
	// Model name.
	Name string
	// Unique id.
	id int
	// State transition probabilities.
	A *narray.NArray
	// Output probabilities.
	B []model.Scorer
	// num states
	ns int
	// Accumulator for transition probabilities.
	TrAcc *narray.NArray
	// Accumulator for global occupation counts.
	OccAcc *narray.NArray
}

// NewNet creates a new HMM network.
func (ms *Set) NewNet(name string, a *narray.NArray, b []model.Scorer) (*Net, error) {

	if len(a.Shape) != 2 || a.Shape[0] != a.Shape[1] {
		panic("rank must be 2 and matrix should be square")
	}
	numStates := a.Shape[0]
	if len(b) != numStates {
		err := fmt.Sprintf("length of b is [%d]. must match shape of a [%d]", len(b), a.Shape[0])
		panic(err)
	}

	net := &Net{
		Name:   name,
		A:      a,
		B:      b,
		TrAcc:  narray.New(numStates, numStates),
		OccAcc: narray.New(numStates),
		ns:     numStates,
	}
	err := ms.add(net)
	if err != nil {
		return nil, err
	}

	return net, nil
}

func (m *Net) logProb(s int, x []float64) float64 {
	o := model.NewIntObs(int(x[0]), model.SimpleLabel(""), "")
	return m.B[s].LogProb(o)
}

func (ms *Set) exist(m *Net) bool {
	if m.id >= len(ms.Nets) || ms.Nets[m.id] != m {
		return false
	}

	return true
}

type chain struct {
	// Composite hmm.
	hmms []*Net
	// Max number of states needed in this chain.
	maxNS int
	// Num hmms in this chain.
	nq int
	// Slice of num states for hmms in this chain.
	ns []int
	// Observations.
	obs model.Obs
	// Observations as float vectors
	vectors [][]float64
	// number of vectors
	nobs int
	// alpha, beta arrays.
	alpha, beta *narray.NArray
	// the model set.
	ms *Set
	// total log prob
	totalProb float64
}

func (ms *Set) chainFromNets(obs model.Obs, m ...*Net) (*chain, error) {

	fos, ok := obs.(model.FloatObsSequence)
	if !ok {
		return nil, fmt.Errorf("obs must be of type model.FloatObsSequence, found type %s which is not supported", reflect.TypeOf(obs.Value()))
	}

	ch := &chain{
		hmms:    m,
		nq:      len(m),
		ns:      make([]int, len(m), len(m)),
		obs:     obs,
		vectors: fos.Value().([][]float64),
		nobs:    len(fos.Value().([][]float64)),
	}

	for k, v := range m {
		if !ms.exist(v) {
			return nil, fmt.Errorf("model [%s] with id [%d] is not in set", v.Name, v.id)
		}
		if v.A.Shape[0] > ch.maxNS {
			ch.maxNS = v.A.Shape[0]
		}
		ch.ns[k] = v.A.Shape[0]
	}
	ch.alpha = narray.New(ch.nq, ch.maxNS, ch.nobs)
	ch.beta = narray.New(ch.nq, ch.maxNS, ch.nobs)

	// Validate.

	// Can't have models with transitions from entry to exit states
	// at the beginning or end of a chain.
	if isTeeModel(m[0]) {
		return nil, fmt.Errorf("the first model in the chain can't have a transition from entry to exit states - model name is [%s] with id [%d]", m[0].Name, m[0].id)
	}
	if isTeeModel(m[len(m)-1]) {
		return nil, fmt.Errorf("the last model in the chain can't have a transition from entry to exit states - model name is [%s] with id [%d]", m[0].Name, m[0].id)
	}

	return ch, nil
}

func (ms *Set) chainFromAssigner(obs model.Obs, assigner Assigner) (*chain, error) {

	// Get the labeler and check that it is of type SimpleLabeler
	// otherwise return error.
	labeler, ok := obs.Label().(model.SimpleLabel)
	if !ok {
		return nil, fmt.Errorf("labeler mas be of type model.SimpleLabel, found type %s which is not supported", reflect.TypeOf(obs.Label()))
	}

	fos, ok := obs.(model.FloatObsSequence)
	if !ok {
		return nil, fmt.Errorf("obs must be of type model.FloatObsSequence, found type %s which is not supported", reflect.TypeOf(obs.Value()))
	}

	// Now we need to assign hmms to the chain.
	// We will use teh sequence of labels to lookup the models by name.
	labels := strings.Split(labeler.String(), ",")
	glog.V(5).Infoln("labeler: ", labeler.String())
	glog.V(5).Infoln("split labels: ", labels)
	if len(labels) == 1 && len(labels[0]) == 0 {
		return nil, fmt.Errorf("no label found, can't assign models")
	}

	modelNames := assigner.Assign(labels)
	glog.V(5).Infoln("labels: ", labels)
	glog.V(5).Infoln("model names: ", modelNames)

	var hmms []*Net
	for _, name := range modelNames {
		h, ok := ms.net(name)
		if !ok {
			return nil, fmt.Errorf("can't find model for name [%s] in set - assignment failed", name)
		}
		hmms = append(hmms, h)
	}
	nq := len(hmms)
	if nq == 0 {
		return nil, fmt.Errorf("the assigner returned no models")
	}

	ch := &chain{
		hmms:    hmms,
		nq:      nq,
		ns:      make([]int, nq, nq),
		obs:     obs,
		vectors: fos.Value().([][]float64),
		nobs:    len(fos.Value().([][]float64)),
	}

	for k, hmm := range hmms {
		if hmm.A.Shape[0] > ch.maxNS {
			ch.maxNS = hmm.A.Shape[0]
		}
		ch.ns[k] = hmm.A.Shape[0]
	}
	ch.alpha = narray.New(ch.nq, ch.maxNS, ch.nobs)
	ch.beta = narray.New(ch.nq, ch.maxNS, ch.nobs)

	// Validate.

	// Can't have models with transitions from entry to exit states
	// at the beginning or end of a chain.
	if isTeeModel(hmms[0]) {
		return nil, fmt.Errorf("the first model in the chain can't have a transition from entry to exit states - model name is [%s] with id [%d]", hmms[0].Name, hmms[0].id)
	}
	if isTeeModel(hmms[nq-1]) {
		return nil, fmt.Errorf("the last model in the chain can't have a transition from entry to exit states - model name is [%s] with id [%d]", hmms[nq-1].Name, hmms[nq-1].id)
	}

	return ch, nil
}

func (ch *chain) update() {

	ch.fb()    // Compute forward-backward probabilities.
	ch.reset() // reset accumulators.
	alpha := ch.alpha
	beta := ch.beta

	logProb := beta.At(0, 0, 0)
	glog.Infof("log prob per observation:%e", logProb/float64(ch.nobs))

	// Compute occupation counts.
	glog.Infof("compute hmm occupation counts.")
	for q, h := range ch.hmms {
		exit := ch.ns[q] - 1
		for t := range ch.vectors {
			for i := 0; i <= exit; i++ {
				v := math.Exp(alpha.At(q, i, t) + beta.At(q, i, t))
				if i == 0 && q < ch.nq-1 {
					// if entry state, add direct trans to next model.
					v += math.Exp(alpha.At(q, 0, t) +
						h.A.At(0, exit) + beta.At(q+1, 0, t))
				}
				h.OccAcc.Inc(v, i)
				glog.V(6).Infof("q:%d, t:%d, i:%d, occ:%e", q, t, i, math.Exp(v))
			}
		}
	}

	// Compute state transition counts.
	glog.Infof("compute hmm state transition counts.")
	for q, h := range ch.hmms {
		ns := ch.ns[q]
		exit := ch.ns[q] - 1
		for t, o := range ch.vectors {
			for i := 0; i < exit; i++ {
				for j := 1; j < ns; j++ {
					var v float64
					switch {
					case i > 0 && j < exit && t < ch.nobs-1:
						// Internal transitions.
						v = alpha.At(q, i, t) + h.A.At(i, j) +
							h.logProb(j, ch.vectors[t+1]) + beta.At(q, j, t+1)
					case i > 0 && j == exit:
						// From internal state to exit state.
						v = alpha.At(q, i, t) + h.A.At(i, exit) + beta.At(q, exit, t)
					case i == 0 && j < exit:
						// From entry state to internal state.
						v = alpha.At(q, 0, t) + h.A.At(0, j) +
							h.logProb(j, o) + beta.At(q, j, t)
					case i == 0 && j == exit && q < ch.nq-1:
						// Direct transition from entry to exit states.
						v = alpha.At(q, 0, t) + h.A.At(0, exit) + beta.At(q+1, 0, t)
					default:
						continue
					}
					h.TrAcc.Inc(math.Exp(v), i, j)
					glog.V(6).Infof("q:%d, t:%d, i:%d, tracc:%f", q, t, i, h.TrAcc.At(i, j))
				}
			}
		}
	}
}

func (ms *Set) reestimate() {

	glog.Infof("reestimate state transition probabilities")
	for id, h := range ms.Nets {
		ns := h.ns
		for i := 0; i < ns; i++ {
			for j := 0; j < ns; j++ {
				v := h.TrAcc.At(i, j) / h.OccAcc.At(i)
				glog.V(5).Infof("id:%d, i:%d, j:%d, old:%f, new:%f",
					id, i, j, math.Exp(h.A.At(i, j)), v)
				h.A.Set(math.Log(v), i, j)
			}
		}
	}
}

func (ch *chain) reset() {

	glog.V(2).Infof("reset accumulators")
	for _, h := range ch.hmms {
		h.OccAcc.SetValue(0.0)
		h.TrAcc.SetValue(0.0)
	}
}

func (ch *chain) fb() {

	nq := ch.nq // num hmm models in chain.
	nobs := ch.nobs

	alpha := ch.alpha
	beta := ch.beta
	hmms := ch.hmms
	ns := ch.ns

	// Compute alpha.

	glog.V(2).Infof("compute forward probabilities")
	alpha.SetValue(math.Inf(-1))

	// t=0, entry state, first model.
	alpha.Set(0.0, 0, 0, 0)
	glog.V(5).Infof("q:%d, i:%d, t:%d, alpha:%f", 0, 0, 0, 0.0)

	// t=0, entry state, after first model.
	for q := 1; q < nq; q++ {
		v := alpha.At(q-1, 0, 0) + hmms[q-1].A.At(0, ns[q-1]-1)
		alpha.Set(v, q, 0, 0)
		glog.V(5).Infof("q:%d, i:%d, t:%d, alpha:%f", q, 0, 0, v)
	}

	// t=0, emitting states.
	for q := 0; q < nq; q++ {
		for j := 1; j < ns[q]-1; j++ {
			v := alpha.At(q, 0, 0) + hmms[q].A.At(0, j) + hmms[q].logProb(j, ch.vectors[0])
			alpha.Set(v, q, j, 0)
			glog.V(5).Infof("q:%d, i:%d, t:%d, alpha:%f", q, j, 0, v)
		}
	}

	// t=0, exit states.
	for q := 0; q < nq; q++ {
		exit := ns[q] - 1
		var v float64
		for i := 1; i < exit; i++ {
			v += math.Exp(alpha.At(q, i, 0) + hmms[q].A.At(i, exit))
		}
		alpha.Set(math.Log(v), q, exit, 0)
		glog.V(5).Infof("q:%d, i:%d, t:%d, alpha:%f", q, exit, 0, math.Log(v))
	}

	for tt := 1; tt < nobs; tt++ {
		for q := 0; q < nq; q++ {
			exit := ns[q] - 1
			for j := 0; j <= exit; j++ {
				var v float64
				switch {
				case j > 0 && j < exit:
					// t>0, emitting states.
					v = math.Exp(alpha.At(q, 0, tt) + hmms[q].A.At(0, j))
					for i := 1; i <= j; i++ {
						v += math.Exp(alpha.At(q, i, tt-1) + hmms[q].A.At(i, j))
					}
					v = math.Log(v) + hmms[q].logProb(j, ch.vectors[tt])
					v = math.Exp(v)
				case j == 0 && q == 0:
					// t>0, entry state, first model.
					// alpha(0,0,t) = -Inf
					v = 0.0
				case j == 0 && q > 0:
					// t>0, entry state, after first model.
					v = math.Exp(alpha.At(q-1, ns[q-1]-1, tt-1)) +
						math.Exp(alpha.At(q-1, 0, tt)+hmms[q-1].A.At(0, ns[q-1]-1))
				case j == exit:
					// t>0, exit states.

					for i := 1; i < exit; i++ {
						v += math.Exp(alpha.At(q, i, tt) + hmms[q].A.At(i, exit))
					}
				}
				alpha.Set(math.Log(v), q, j, tt)
				glog.V(5).Infof("q:%d, i:%d, t:%d, alpha:%f", q, j, tt, math.Log(v))
			}
		}
	}

	alphaLogProb := alpha.At(nq-1, ch.ns[nq-1]-1, nobs-1)
	glog.V(2).Infof("alpha total prob:%f, avg per obs:%f",
		alphaLogProb, alphaLogProb/float64(nobs))

	// Compute beta.

	beta.SetValue(math.Inf(-1))

	// t=nobs-1, exit state, last model.
	beta.Set(0, nq-1, ns[nq-1]-1, nobs-1)
	glog.V(5).Infof("q:%d, i:%d, t:%d,  beta:%f", nq-1, ns[nq-1]-1, nobs-1, 0.0)

	// t=nobs-1, exit state, before last model.
	for q := nq - 2; q >= 0; q-- {
		v := beta.At(q+1, ns[q+1]-1, nobs-1) + hmms[q+1].A.At(0, ns[q+1]-1)
		beta.Set(v, q, ns[q]-1, nobs-1)
		glog.V(5).Infof("q:%d, i:%d, t:%d,  beta:%f", q, ns[q]-1, nobs-1, v)
	}

	// t=nobs-1, emitting states.
	for q := nq - 1; q >= 0; q-- {
		for i := ns[q] - 2; i > 0; i-- {
			v := hmms[q].A.At(i, ns[q]-1) + beta.At(q, ns[q]-1, nobs-1)
			beta.Set(v, q, i, nobs-1)
			glog.V(5).Infof("q:%d, i:%d, t:%d,  beta:%f", q, i, nobs-1, v)
		}
	}

	// t=nobs-1, entry states.
	for q := nq - 1; q >= 0; q-- {
		var v float64
		for j := 1; j < ns[q]-1; j++ {
			v += math.Exp(hmms[q].A.At(0, j) +
				hmms[q].logProb(j, ch.vectors[nobs-1]) + beta.At(q, j, nobs-1))
		}
		beta.Set(math.Log(v), q, 0, nobs-1)
		glog.V(5).Infof("q:%d, i:%d, t:%d,  beta:%f", q, 0, nobs-1, math.Log(v))
	}

	for tt := nobs - 2; tt >= 0; tt-- {
		for q := nq - 1; q >= 0; q-- {
			exit := ns[q] - 1
			for i := exit; i >= 0; i-- {
				var v float64
				switch {
				case i > 0 && i < exit:
					// t<nobs-1, emitting states.
					v = math.Exp(hmms[q].A.At(i, exit) + beta.At(q, exit, tt))
					for j := i; j < exit; j++ {
						v += math.Exp(hmms[q].A.At(i, j) +
							hmms[q].logProb(j, ch.vectors[tt+1]) + beta.At(q, j, tt+1))
					}
				case i == exit && q == nq-1:
					// t<nobs-1, exit state, last model.
					v = 0.0
				case i == exit && q < nq-1:
					// t<nobs-1, exit state, before last model.
					v = math.Exp(beta.At(q+1, 0, tt+1)) +
						math.Exp(beta.At(q+1, ns[q+1]-1, tt)+
							hmms[q+1].A.At(0, ns[q+1]-1))
				case i == 0:
					// t<nobs-1, entry states.
					for j := 1; j < exit; j++ {
						v += math.Exp(hmms[q].A.At(0, j) +
							hmms[q].logProb(j, ch.vectors[tt]) + beta.At(q, j, tt))
					}
				}
				beta.Set(math.Log(v), q, i, tt)
				glog.V(5).Infof("q:%d, i:%d, t:%d,  beta:%f", q, i, tt, math.Log(v))
			}
		}
	}

	betaLogProb := beta.At(0, 0, 0)
	glog.V(4).Infof("beta total prob:%f, avg per obs:%f", betaLogProb, betaLogProb/float64(nobs))

	diff := (alphaLogProb - betaLogProb) / float64(nobs)
	glog.V(4).Infof("alpha-beta relative diff:%f", diff)

	return
}

func isTeeModel(m *Net) bool {

	exit := m.ns - 1
	if m.A.At(0, exit) > math.Inf(-1) {
		return true
	}
	return false
}

// makes left-to-right hmm with self loops.
// ns is the total number of states including entry/exit.
// selfProb is the prob of the self loop with value between 0 and 1.
// skipProb is the prob of skipping next state. Make it zero for no skips.
func (ms *Set) makeLetfToRight(name string, ns int, selfProb,
	skipProb float64, dists []model.Scorer) (*Net, error) {

	if selfProb >= 1 || skipProb >= 1 || selfProb < 0 || skipProb < 0 {
		panic("probabilities must have value >= 0 and < 1")
	}
	if selfProb+skipProb >= 1 {
		panic("selfProb + skipProb must be less than 1")
	}
	if len(dists) != ns {
		panic("length of dists must match number of states")
	}
	hmm, err := ms.NewNet(name, narray.New(ns, ns), dists)
	if err != nil {
		return nil, err
	}
	p := selfProb
	q := skipProb
	r := 1.0 - p - q

	// state 0
	hmm.A.Set(1-q, 0, 1) // entry
	hmm.A.Set(q, 0, 2)   // skip first emmiting state

	// states 1..ns-3
	for i := 1; i < ns-2; i++ {
		hmm.A.Set(p, i, i)   // self loop
		hmm.A.Set(r, i, i+1) // to right
		hmm.A.Set(q, i, i+2) // skip
	}
	// state ns-2
	hmm.A.Set(p, ns-2, ns-2)   // self
	hmm.A.Set(1-p, ns-2, ns-1) // to exit (no skip)

	// convert to log.
	hmm.A = narray.Log(nil, hmm.A.Copy())
	return hmm, nil
}
