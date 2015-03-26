// Copyright (c) 2015 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

import (
	"fmt"
	"math"
	"math/rand"
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

	if glog.V(3) {
		ns := m.ns
		for i := 0; i < ns; i++ {
			for j := 0; j < ns; j++ {
				p := math.Exp(m.A.At(i, j))
				if p > smallNumber {
					glog.Infof("added A id:%d, name:%s, from:%d, to:%d, prob:%5.3f", m.id, m.Name, i, j, p)
				}
			}
		}
	}
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

func (ms *Set) reset() {
	glog.V(2).Infof("reset accumulators")
	for _, h := range ms.Nets {
		h.OccAcc.SetValue(0.0)
		h.TrAcc.SetValue(0.0)
		for i := 1; i < h.ns-1; i++ {
			h.B[i].Clear()
		}
	}
}

func (ms *Set) size() int {
	return len(ms.Nets)
}

// Net is an hmm network with a single non-emmiting entry state (index 0) and
// a single non-emmiting exit state (index ns-1) where ns is the total number
// of states. The HMMs must have a left-to-right topology. That is, transitions
// can only go from state i to state j where j >= i.
type Net struct {
	// Model name.
	Name string
	// num states
	ns int
	// Unique id.
	id int
	// State transition probabilities. (ns x ns matrix)
	A *narray.NArray
	// Output probabilities. (ns x 1 vector)
	B []model.Modeler
	// Accumulator for transition probabilities.
	TrAcc *narray.NArray
	// Accumulator for global occupation counts.
	OccAcc *narray.NArray
}

// NewNet creates a new HMM network.
func (ms *Set) NewNet(name string, a *narray.NArray, b []model.Modeler) (*Net, error) {

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
	o := model.NewFloatObs(x, model.SimpleLabel(""))
	return m.B[s].LogProb(o)
}

func (ms *Set) exist(m *Net) bool {
	if m.id >= len(ms.Nets) || ms.Nets[m.id] != m {
		return false
	}
	return true
}

func (m *Net) nextState(s int, r *rand.Rand) int {

	dist := m.A.SubArray(s, -1)
	return model.RandIntFromLogDist(dist.Data, r)
}

// chain is used to concatenate hmms during training based on the labels.
// For example, in speech a chain would represent a sequence of words based
// on an orthographic transcription of the training utterance.
type chain struct {
	// Composite hmm.
	hmms []*Net
	// Max number of states needed in this chain.
	maxNS int
	// Num hmms in this chain.
	nq int
	// Num states in hmms.
	ns []int
	// Observation.
	obs model.Obs
	// Likelihoods p(obs/state)
	likelihoods *narray.NArray
	// Raw observation data for sequence.
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

	if glog.V(6) {
		glog.Info("lab: ", ch.obs.Label())
		glog.Info("obs: ", ch.obs.Value())
	}

	ch.computeLikelihoods()

	// Validate.

	// Can't have models with transitions from entry to exit states
	// at the beginning or end of a chain.
	if isTeeModel(m[0]) {
		return nil, fmt.Errorf("first model in chain can't have entry to exit transition - model name is [%s] with id [%d]", m[0].Name, m[0].id)
	}
	if isTeeModel(m[len(m)-1]) {
		return nil, fmt.Errorf("last model in chain can't have entry to exit transition - model name is [%s] with id [%d]", m[0].Name, m[0].id)
	}

	return ch, nil
}

// Use the label to create a chain of hmms.
// If model set only has one hmm, no need to use labels, simply assign the only hmm.
func (ms *Set) chainFromAssigner(obs model.Obs, assigner Assigner) (*chain, error) {

	var hmms []*Net
	var fos model.FloatObsSequence
	switch o := obs.(type) {
	case model.FloatObsSequence:
		fos = o
	case *model.FloatObsSequence:
		fos = *o
	default:
		return nil, fmt.Errorf("obs must be of type model.FloatObsSequence, found type %s which is not supported",
			reflect.TypeOf(obs))
	}

	if assigner == nil && ms.size() == 1 {
		hmms = append(hmms, ms.Nets[0])
		if glog.V(3) {
			glog.Warningf("assigner missing but model set has only one hmm network - assigning model [%s] to chain", ms.Nets[0].Name)
		}
	} else if assigner == nil {
		return nil, fmt.Errorf("need assigner to create hmm chain if model set is greater than one")
	} else {
		// Get the labeler and check that it is of type SimpleLabeler
		// otherwise return error.
		labeler, ok := obs.Label().(model.SimpleLabel)
		if !ok {
			return nil, fmt.Errorf("labeler mas be of type model.SimpleLabel, found type %s which is not supported",
				reflect.TypeOf(obs.Label()))
		}
		// Now we need to assign hmms to the chain.
		// We will use teh sequence of labels to lookup the models by name.
		labels := strings.Split(labeler.String(), ",")
		if len(labels) == 1 && len(labels[0]) == 0 {
			return nil, fmt.Errorf("no label found, can't assign models")
		}
		modelNames := assigner.Assign(labels)
		glog.V(5).Infoln("labels: ", labels)
		glog.V(5).Infoln("model names: ", modelNames)

		for _, name := range modelNames {
			h, ok := ms.net(name)
			if !ok {
				return nil, fmt.Errorf("can't find model for name [%s] in set - assignment failed", name)
			}
			hmms = append(hmms, h)
		}
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

	if glog.V(6) {
		glog.Info("lab: ", ch.obs.Label())
		glog.Info("obs: ", ch.obs.Value())
	}

	ch.computeLikelihoods()

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

func (ch *chain) computeLikelihoods() {

	ch.likelihoods = narray.New(ch.nq, ch.maxNS, ch.nobs)
	for q, h := range ch.hmms {
		for i := 1; i < ch.ns[q]-1; i++ {
			for t, v := range ch.vectors {
				ll := h.logProb(i, v)
				ch.likelihoods.Set(ll, q, i, t)
				glog.V(6).Infof("q:%d, i:%d, t:%d, likelihood:%8.3f", q, i, t, ll)
			}
		}
	}
}

func (ch *chain) update() error {

	ch.fb() // Compute forward-backward probabilities.

	logProb := ch.beta.At(0, 0, 0)
	totalProb := math.Exp(logProb)
	if logProb == math.Inf(-1) {
		return fmt.Errorf("log prob is -Inf, skipping training sequence")
	}
	glog.V(2).Infof("log prob per observation:%e", logProb/float64(ch.nobs))

	// Compute state transition counts.
	glog.V(2).Infof("compute hmm state transition counts.")
	for q, h := range ch.hmms {
		exit := ch.ns[q] - 1
		for t, vec := range ch.vectors {
			for i := 0; i < exit; i++ {
				w := ch.doOccAcc(q, i, t, totalProb) / totalProb
				ch.doTrAcc(q, i, t, totalProb)
				if i > 0 {
					o := model.F64ToObs(vec, "")
					h.B[i].UpdateOne(o, w) // TODO prove!
				}
			}
		}
	}
	return nil
}

func (ch *chain) doOccAcc(q, i, t int, tp float64) float64 {

	h := ch.hmms[q]
	exit := ch.ns[q] - 1
	vv := ch.alpha.At(q, i, t) + ch.beta.At(q, i, t)
	v := math.Exp(vv)

	if i == 0 && q < ch.nq-1 {
		// if entry state, add direct trans to next model.
		w := ch.alpha.At(q, 0, t) + h.A.At(0, exit) + ch.beta.At(q+1, 0, t)
		v += math.Exp(w)
		if w > vv {
			vv = w
		}
	}
	h.OccAcc.Inc(v/tp, i)
	glog.V(6).Infof("q:%d, t:%d, i:%d, occ:%.0f", q, t, i, vv)
	return v
}

func (ch *chain) doTrAcc(q, i, t int, tp float64) {

	h := ch.hmms[q]
	ns := ch.ns[q]
	exit := ch.ns[q] - 1
	//	op0 := ch.vectors[t]
	for j := 1; j < ns; j++ {
		var v float64
		switch {
		case i > 0 && j < exit && t < ch.nobs-1:
			// Internal transitions.
			v = ch.alpha.At(q, i, t) + h.A.At(i, j) +
				ch.likelihoods.At(q, j, t+1) + ch.beta.At(q, j, t+1)
		case i > 0 && j == exit:
			// From internal state to exit state.
			v = ch.alpha.At(q, i, t) + h.A.At(i, exit) + ch.beta.At(q, exit, t)
		case i == 0 && j < exit:
			// From entry state to internal state.
			v = ch.alpha.At(q, 0, t) + h.A.At(0, j) +
				ch.likelihoods.At(q, j, t) + ch.beta.At(q, j, t)
		case i == 0 && j == exit && q < ch.nq-1:
			// Direct transition from entry to exit states.
			v = ch.alpha.At(q, 0, t) + h.A.At(0, exit) + ch.beta.At(q+1, 0, t)
		default:
			continue
		}
		h.TrAcc.Inc(math.Exp(v)/tp, i, j)
		glog.V(6).Infof("q:%d, t:%d, i:%d, j:%d, tracc:%.0f", q, t, i, j, v)
	}
}

func (ms *Set) reestimate() {

	glog.V(2).Infof("reestimate state transition probabilities")
	for _, h := range ms.Nets {
		ns := h.ns
		for i := 0; i < ns; i++ {
			if i > 0 && i < ns-1 {
				err := h.B[i].Estimate()
				if err != nil {
					glog.Errorf("model estimation error: %s", err)
				}
			}
			for j := i; j < ns && i < ns-1; j++ {
				v := h.TrAcc.At(i, j) / h.OccAcc.At(i)
				h.A.Set(math.Log(v), i, j)
				glog.V(6).Infof("reest add A %d=>%d, tracc:%6.4f, occacc:%6.4f, p:%5.3f", i, j, h.TrAcc.At(i, j), h.OccAcc.At(i), v)
				if math.IsNaN(h.A.At(i, j)) {
					panic("reestimated transition prob is NaN")
				}
			}
		}
	}

	// Traces.
	if glog.V(3) {
		for id, h := range ms.Nets {
			ns := h.ns
			for i := 0; i < ns; i++ {
				glog.Infof("i:%d, total_occacc:%e (%.0f)",
					i, h.OccAcc.At(i), math.Log(h.OccAcc.At(i)))
				for j := 0; j < ns; j++ {
					glog.Infof("i:%d, j:%d, total_tracc:%e (%.0f)",
						i, j, h.TrAcc.At(i, j), math.Log(h.TrAcc.At(i, j)))
					p := math.Exp(h.A.At(i, j))
					if p > smallNumber {
						glog.Infof("reest A id:%d, name:%s, from:%d, to:%d, prob:%5.2f",
							id, h.Name, i, j, p)
					}
				}
			}
		}
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
	glog.V(5).Infof("q:%d, i:%d, t:%d, alpha:%.0f", 0, 0, 0, 0.0)

	// t=0, entry state, after first model.
	for q := 1; q < nq; q++ {
		v := alpha.At(q-1, 0, 0) + hmms[q-1].A.At(0, ns[q-1]-1)
		alpha.Set(v, q, 0, 0)
		glog.V(5).Infof("q:%d, i:%d, t:%d, alpha:%.0f", q, 0, 0, v)
	}

	// t=0, emitting states.
	for q := 0; q < nq; q++ {
		for j := 1; j < ns[q]-1; j++ {
			v := alpha.At(q, 0, 0) + hmms[q].A.At(0, j) + ch.likelihoods.At(q, j, 0)
			alpha.Set(v, q, j, 0)
			glog.V(5).Infof("q:%d, j:%d, t:%d, alpha:%.0f", q, j, 0, v)
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
		glog.V(5).Infof("q:%d, i:%d, t:%d, alpha:%.0f", q, exit, 0, math.Log(v))
	}

	for tt := 1; tt < nobs; tt++ {
		for q := 0; q < nq; q++ {
			exit := ns[q] - 1
			for j := 0; j <= exit; j++ {
				var v float64
				switch {
				case j > 0 && j < exit:
					// t>0, emitting states.
					w := math.Exp(alpha.At(q, 0, tt) + hmms[q].A.At(0, j))
					for i := 1; i <= j; i++ {
						w += math.Exp(alpha.At(q, i, tt-1) + hmms[q].A.At(i, j))
					}
					v = math.Log(w) + ch.likelihoods.At(q, j, tt)
					v = math.Exp(v)
				case j == 0 && q == 0:
					// t>0, entry state, first model.
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
				glog.V(5).Infof("q:%d, i:%d, t:%d, alpha:%.0f", q, j, tt, math.Log(v))
			}
		}
	}

	alphaLogProb := alpha.At(nq-1, ch.ns[nq-1]-1, nobs-1)
	glog.V(2).Infof("alpha total prob:%.0f, avg per obs:%.0f",
		alphaLogProb, alphaLogProb/float64(nobs))

	// Compute beta.

	beta.SetValue(math.Inf(-1))

	// t=nobs-1, exit state, last model.
	beta.Set(0, nq-1, ns[nq-1]-1, nobs-1)
	glog.V(5).Infof("q:%d, i:%d, t:%d,  beta:%.0f", nq-1, ns[nq-1]-1, nobs-1, 0.0)

	// t=nobs-1, exit state, before last model.
	for q := nq - 2; q >= 0; q-- {
		v := beta.At(q+1, ns[q+1]-1, nobs-1) + hmms[q+1].A.At(0, ns[q+1]-1)
		beta.Set(v, q, ns[q]-1, nobs-1)
		glog.V(5).Infof("q:%d, i:%d, t:%d,  beta:%.0f", q, ns[q]-1, nobs-1, v)
	}

	// t=nobs-1, emitting states.
	for q := nq - 1; q >= 0; q-- {
		for i := ns[q] - 2; i > 0; i-- {
			v := hmms[q].A.At(i, ns[q]-1) + beta.At(q, ns[q]-1, nobs-1)
			beta.Set(v, q, i, nobs-1)
			glog.V(5).Infof("q:%d, i:%d, t:%d,  beta:%.0f", q, i, nobs-1, v)
		}
	}

	// t=nobs-1, entry states.
	for q := nq - 1; q >= 0; q-- {
		var v float64
		for j := 1; j < ns[q]-1; j++ {
			v += math.Exp(hmms[q].A.At(0, j) +
				ch.likelihoods.At(q, j, nobs-1) + beta.At(q, j, nobs-1))

		}
		beta.Set(math.Log(v), q, 0, nobs-1)
		glog.V(5).Infof("q:%d, i:%d, t:%d,  beta:%.0f", q, 0, nobs-1, math.Log(v))
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
							ch.likelihoods.At(q, j, tt+1) + beta.At(q, j, tt+1))
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
							ch.likelihoods.At(q, j, tt) + beta.At(q, j, tt))
					}
				}
				beta.Set(math.Log(v), q, i, tt)
				glog.V(5).Infof("q:%d, i:%d, t:%d,  beta:%.0f", q, i, tt, math.Log(v))
			}
		}
	}

	betaLogProb := beta.At(0, 0, 0)
	glog.V(2).Infof("beta total prob:%.0f, avg per obs:%.0f", betaLogProb, betaLogProb/float64(nobs))

	diff := (alphaLogProb - betaLogProb) / float64(nobs)
	glog.V(2).Infof("alpha-beta relative diff:%e", diff)

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
func (ms *Set) makeLeftToRight(name string, ns int, selfProb,
	skipProb float64, dists []model.Modeler) (*Net, error) {

	if selfProb >= 1 || skipProb >= 1 || selfProb < 0 || skipProb < 0 {
		panic("probabilities must have value >= 0 and < 1")
	}
	if selfProb+skipProb >= 1 {
		panic("selfProb + skipProb must be less than 1")
	}
	if len(dists) != ns {
		panic("length of dists must match number of states")
	}

	p := selfProb
	q := skipProb
	r := 1.0 - p - q

	h := narray.New(ns, ns)

	// state 0
	h.Set(1-q, 0, 1) // entry
	h.Set(q, 0, 2)   // skip first emmiting state

	// states 1..ns-3
	for i := 1; i < ns-2; i++ {
		h.Set(p, i, i)   // self loop
		h.Set(r, i, i+1) // to right
		h.Set(q, i, i+2) // skip
	}
	// state ns-2
	h.Set(p, ns-2, ns-2)   // self
	h.Set(1-p, ns-2, ns-1) // to exit (no skip)

	// convert to log.
	h = narray.Log(nil, h.Copy())

	hmm, err := ms.NewNet(name, h, dists)
	if err != nil {
		return nil, err
	}
	return hmm, nil
}

// MakeLeftToRight creates a transition probability matrix for a left-to-right HMM.
// ns is the total number of states including entry/exit.
// selfProb is the prob of the self loop with value between 0 and 1.
// skipProb is the prob of skipping next state. Make it zero for no skips.
func MakeLeftToRight(ns int, selfProb, skipProb float64) *narray.NArray {

	if selfProb >= 1 || skipProb >= 1 || selfProb < 0 || skipProb < 0 {
		panic("probabilities must have value >= 0 and < 1")
	}
	if selfProb+skipProb >= 1 {
		panic("selfProb + skipProb must be less than 1")
	}
	p := selfProb
	q := skipProb
	r := 1.0 - p - q

	h := narray.New(ns, ns)

	// state 0
	h.Set(1-q, 0, 1) // entry
	h.Set(q, 0, 2)   // skip first emmiting state

	// states 1..ns-3
	for i := 1; i < ns-2; i++ {
		h.Set(p, i, i)   // self loop
		h.Set(r, i, i+1) // to right
		h.Set(q, i, i+2) // skip
	}
	// state ns-2
	h.Set(p, ns-2, ns-2)   // self
	h.Set(1-p, ns-2, ns-1) // to exit (no skip)

	return narray.Log(h, h)
}
