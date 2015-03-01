package hmm

import (
	"fmt"
	"math"

	"github.com/akualab/gjoa/model"
	"github.com/akualab/narray"
	"github.com/golang/glog"
)

// The actual hmm network implementation.

type modelSet map[uint16]*hmm

func (ms modelSet) add(hmms []*hmm) {

	for _, h := range hmms {
		_, exist := ms[h.id]
		if !exist {
			ms[h.id] = h
			glog.V(1).Infof("add model [%s] with id [%d] to set", h.name, h.id)
		}
	}
}

type hmm struct {
	// Model name.
	name string
	// Unique id.
	id uint16
	// State transition probabilities.
	a *narray.NArray
	// Output probabilities.
	b []model.Scorer
	// num states
	ns int
	// Accumulator for transition probabilities.
	trAcc *narray.NArray
	// Accumulator for global occupation counts.
	occAcc *narray.NArray
}

func newHMM(name string, id uint16, a *narray.NArray, b []model.Scorer) *hmm {

	if len(a.Shape) != 2 || a.Shape[0] != a.Shape[1] {
		panic("rank must be 2 and matrix should be square")
	}
	numStates := a.Shape[0]
	if len(b) != numStates {
		err := fmt.Sprintf("length of b is [%d]. must match shape of a [%d]", len(b), a.Shape[0])
		panic(err)
	}

	return &hmm{
		name:   name,
		id:     id,
		a:      a,
		b:      b,
		trAcc:  narray.New(numStates, numStates),
		occAcc: narray.New(numStates),
		ns:     numStates,
	}
}

func (m *hmm) logProb(s int, x model.Obs) float64 {
	return m.b[s].LogProb(x)
}

type chain struct {
	// Composite hmm.
	hmms []*hmm
	// Max number of states needed in this chain.
	maxNS int
	// Num hmms in this chain.
	nq int
	// Slice of num states for hmms in this chain.
	ns []int
	// Observations.
	obs []model.Obs
	// alpha, beta arrays.
	alpha, beta *narray.NArray
	// the model set.
	ms modelSet
}

func newChain(ms modelSet, obs []model.Obs, m ...*hmm) *chain {

	if ms == nil {
		panic("model set must be initialized")
	}
	ms.add(m) // add models to set
	ch := &chain{
		hmms: m,
		nq:   len(m),
		ns:   make([]int, len(m), len(m)),
		obs:  obs,
	}

	for k, v := range m {
		if v.a.Shape[0] > ch.maxNS {
			ch.maxNS = v.a.Shape[0]
		}
		ch.ns[k] = v.a.Shape[0]
	}
	nobs := len(obs)
	ch.alpha = narray.New(ch.nq, ch.maxNS, nobs)
	ch.beta = narray.New(ch.nq, ch.maxNS, nobs)
	return ch
}

func (ch *chain) update() {

	ch.fb()    // Compute forward-backward probabilities.
	ch.reset() // reset accumulators.
	nobs := len(ch.obs)
	alpha := ch.alpha
	beta := ch.beta

	logProb := beta.At(0, 0, 0)
	glog.Infof("log prob per observation:%e", logProb/float64(nobs))

	// Compute occupation counts.
	glog.Infof("compute hmm occupation counts.")
	for q, h := range ch.hmms {
		last := ch.ns[q] - 1
		for t := range ch.obs {
			for i := 0; i <= last; i++ {
				v := math.Exp(alpha.At(q, i, t) + beta.At(q, i, t))
				if i == 0 && q < ch.nq-1 {
					// if entry state, add direct trans to next model.
					v += math.Exp(alpha.At(q, 0, t) +
						h.a.At(0, last) + beta.At(q+1, 0, t))
				}
				h.occAcc.Inc(v, i)
				glog.V(4).Infof("q:%d, t:%d, i:%d, occ:%e", q, t, i, math.Exp(v))
			}
		}
	}

	// Compute state transition counts.
	glog.Infof("compute hmm state transition counts.")
	for q, h := range ch.hmms {
		ns := ch.ns[q]
		last := ch.ns[q] - 1
		for t, o := range ch.obs {
			for i := 0; i < last; i++ {
				for j := 1; j < ns; j++ {

					if i == 0 && j < last {

						// From entry state to internal state.
						v := alpha.At(q, 0, t) + h.a.At(0, j) +
							h.logProb(j, o) + beta.At(q, j, t)
						h.trAcc.Inc(math.Exp(v), 0, j)

					} else if i > 0 && j < last && t < nobs-1 {

						// Internal transitions.
						v := alpha.At(q, i, t) + h.a.At(i, j) +
							h.logProb(j, ch.obs[t+1]) + beta.At(q, j, t+1)
						h.trAcc.Inc(math.Exp(v), i, j)

					} else if i > 0 && j == last {

						// From internal state to exit state.
						v := alpha.At(q, i, t) + h.a.At(i, last) + beta.At(q, last, t)
						h.trAcc.Inc(math.Exp(v), i, last)
					}

					// Direct transition from entry to exit states.
					if i == 0 && j == last && q < ch.nq-1 {
						v := alpha.At(q, 0, t) + h.a.At(0, last) + beta.At(q+1, 0, t)
						h.trAcc.Inc(math.Exp(v), 0, last)
					}
					glog.V(4).Infof("q:%d, t:%d, i:%d, tracc:%e", q, t, i, h.trAcc.At(i, j))
				}
			}
		}
	}
}

func (ms modelSet) reestimate() {

	glog.Infof("reestimate state transition probabilities")
	for id, h := range ms {
		ns := h.ns
		for i := 0; i < ns; i++ {
			for j := 0; j < ns; j++ {
				v := h.trAcc.At(i, j) / h.occAcc.At(i)
				glog.V(4).Infof("name:%10s, id:%d, i:%d, j:%d, old:%e, new:%e", h.name, id, i, j,
					math.Exp(h.a.At(i, j)), v)
				h.a.Set(math.Log(v), i, j)
			}
		}
	}
}

func (ch *chain) reset() {

	glog.V(2).Infof("reset accumulators")
	for _, h := range ch.hmms {
		h.occAcc.SetValue(0.0)
		h.trAcc.SetValue(0.0)
	}
}

func (ch *chain) fb() {

	nq := ch.nq         // num hmm models in chain.
	nobs := len(ch.obs) // Num observations.

	alpha := ch.alpha
	beta := ch.beta
	hmms := ch.hmms
	nstates := ch.ns

	// Compute alpha.

	glog.V(2).Infof("compute forward probabilities")
	alpha.SetValue(math.Inf(-1))

	// t=0, entry state, first model.
	alpha.Set(0.0, 0, 0, 0)

	// t=0, entry state, after first model.
	for q := 1; q < nq; q++ {
		v := alpha.At(q-1, 0, 0) + hmms[q-1].a.At(0, nstates[q-1]-1)
		alpha.Set(v, q, 0, 0)
	}

	// t=0, emitting states.
	for q := 0; q < nq; q++ {
		for j := 1; j < nstates[q]-1; j++ {
			v := hmms[q].a.At(0, j) + hmms[q].logProb(j, ch.obs[0])
			alpha.Set(v, q, j, 0)
		}
	}

	// t=0, exit states.
	for q := 0; q < nq; q++ {
		var v float64
		for i := 1; i < nstates[q]-1; i++ {
			v += math.Exp(alpha.At(q, i, 0) + hmms[q].a.At(i, nstates[q]-1))
		}
		alpha.Set(math.Log(v), q, nstates[q]-1, 0)
	}

	for q := 0; q < nq; q++ {
		for tt := 1; tt < nobs; tt++ {
			if q == 0 {
				// t>0, entry state, first model.
				// alpha(0,0,t) = 0
			} else {
				// t>0, entry state, after first model.
				v := math.Exp(alpha.At(q-1, nstates[q-1]-1, tt-1)) + math.Exp(alpha.At(q-1, 0, tt)+hmms[q-1].a.At(0, nstates[q-1]-1))
				alpha.Set(math.Log(v), q, 0, tt)
			}

			// t>0, emitting states.
			for j := 1; j < nstates[q]-1; j++ {
				v := math.Exp(alpha.At(q, 0, tt) + hmms[q].a.At(0, j))
				for i := 1; i <= j; i++ {
					v += math.Exp(alpha.At(q, i, tt-1) + hmms[q].a.At(i, j))
				}
				v = math.Log(v) + hmms[q].logProb(j, ch.obs[tt])
				alpha.Set(v, q, j, tt)
			}

			// t>0, exit states.
			var v float64
			for i := 1; i < nstates[q]-1; i++ {
				v += math.Exp(alpha.At(q, i, tt) + hmms[q].a.At(i, nstates[q]-1))
			}
			alpha.Set(math.Log(v), q, nstates[q]-1, tt)
		}
	}

	alphaLogProb := alpha.At(nq-1, ch.ns[nq-1]-1, nobs-1)
	glog.V(2).Infof("alpha total prob:%f, avg per obs:%f", alphaLogProb, alphaLogProb/float64(nobs))

	// Compute beta.

	beta.SetValue(math.Inf(-1))

	// t=nobs-1, exit state, last model.
	beta.Set(0, nq-1, nstates[nq-1]-1, nobs-1)

	// t=nobs-1, exit state, before last model.
	for q := nq - 2; q >= 0; q-- {
		v := beta.At(q+1, nstates[q+1]-1, nobs-1) + hmms[q+1].a.At(0, nstates[q+1]-1)
		beta.Set(v, q, nstates[q]-1, nobs-1)
	}

	// t=nobs-1, emitting states.
	for q := nq - 1; q >= 0; q-- {
		for i := nstates[q] - 2; i > 0; i-- {
			v := hmms[q].a.At(i, nstates[q]-1) + beta.At(q, nstates[q]-1, nobs-1)
			beta.Set(v, q, i, nobs-1)
		}
	}

	// t=nobs-1, entry states.
	for q := nq - 1; q >= 0; q-- {
		var v float64
		for j := 1; j < nstates[q]-1; j++ {
			v += math.Exp(hmms[q].a.At(0, j) + hmms[q].logProb(j, ch.obs[nobs-1]) + beta.At(q, j, nobs-1))
		}
		beta.Set(math.Log(v), q, 0, nobs-1)
	}

	for q := nq - 1; q >= 0; q-- {
		for tt := nobs - 2; tt >= 0; tt-- {

			if q == nq-1 {
				// t<nobs-1, exit state, last model.
			} else {
				// t<nobs-1, exit state, before last model.
				v := math.Exp(beta.At(q+1, 0, tt+1)) + math.Exp(beta.At(q+1, nstates[q+1]-1, tt)+hmms[q+1].a.At(0, nstates[q+1]-1))
				beta.Set(math.Log(v), q, nstates[q]-1, tt)
			}
			// t<nobs-1, emitting states.
			for i := nstates[q] - 2; i > 0; i-- {
				v := math.Exp(hmms[q].a.At(i, nstates[q]-1) + beta.At(q, nstates[q]-1, tt))
				for j := i; j < nstates[q]-1; j++ {
					v += math.Exp(hmms[q].a.At(i, j) + hmms[q].logProb(j, ch.obs[tt+1]) + beta.At(q, j, tt+1))
				}
				beta.Set(math.Log(v), q, i, tt)
			}

			// t<nobs-1, entry states.
			var v float64
			for j := 1; j < nstates[q]-1; j++ {
				v += math.Exp(hmms[q].a.At(0, j) + hmms[q].logProb(j, ch.obs[tt]) + beta.At(q, j, tt))
			}
			beta.Set(math.Log(v), q, 0, tt)
		}
	}

	betaLogProb := beta.At(0, 0, 0)
	glog.V(4).Infof("beta total prob:%f, avg per obs:%f", betaLogProb, betaLogProb/float64(nobs))

	diff := (alphaLogProb - betaLogProb) / float64(nobs)
	glog.V(4).Infof("alpha-beta relative diff:%e", diff)

	return
}
