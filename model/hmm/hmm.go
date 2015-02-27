package hmm

import (
	"fmt"
	"math"

	"github.com/akualab/gjoa/model"
	"github.com/akualab/narray"
)

// The actual hmm network implementation.

type hmm struct {
	name string
	id   uint16
	a    *narray.NArray
	b    []model.Scorer
}

func newHMM(name string, id uint16, a *narray.NArray, b []model.Scorer) *hmm {

	if len(a.Shape) != 2 || a.Shape[0] != a.Shape[1] {
		panic("rank of a must be 2 and matrix should be square")
	}
	if a.Shape[0] != len(b) {
		err := fmt.Sprintf("length of b is [%d]. must match shape of a [%d]", len(b), a.Shape[0])
		panic(err)
	}

	return &hmm{
		name: name,
		id:   id,
		a:    a,
		b:    b,
	}
}

func (m *hmm) logProb(s int, x model.Obs) float64 {
	return m.b[s].LogProb(x)
}

type chain struct {
	hmms  []*hmm
	maxNS int
	nq    int
	ns    []int
}

func newChain(m ...*hmm) *chain {

	ch := &chain{
		hmms: m,
		nq:   len(m),
		ns:   make([]int, len(m), len(m)),
	}

	for k, v := range m {
		if v.a.Shape[0] > ch.maxNS {
			ch.maxNS = v.a.Shape[0]
		}
		ch.ns[k] = v.a.Shape[0]
	}
	return ch
}

func (ch *chain) first() *hmm {
	return ch.hmms[0]
}

func (ch *chain) last() *hmm {
	return ch.hmms[ch.nq-1]
}

func (ch *chain) fb(obs []model.Obs) (alpha *narray.NArray, beta *narray.NArray) {

	nq := ch.nq      // num hmm models in chain.
	nobs := len(obs) // Num observations.

	alpha = narray.New(nq, ch.maxNS, nobs)
	beta = narray.New(nq, ch.maxNS, nobs)
	hmms := ch.hmms
	nstates := ch.ns

	// Compute alpha.

	alpha.SetValue(math.Inf(-1))
	beta.SetValue(math.Inf(-1))

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
			v := hmms[q].a.At(0, j) + hmms[q].logProb(j, obs[0])
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
				for i := 1; i < nstates[q]-1; i++ {
					v += math.Exp(alpha.At(q, i, tt-1) + hmms[q].a.At(i, j))
				}
				v = math.Log(v) + hmms[q].logProb(j, obs[tt])
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

	// Compute beta.

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
			v += math.Exp(hmms[q].a.At(0, j) + hmms[q].logProb(j, obs[nobs-1]) + beta.At(q, j, nobs-1))
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
				for j := 1; j < nstates[q]-1; j++ {
					v += math.Exp(hmms[q].a.At(i, j) + hmms[q].logProb(j, obs[tt+1]) + beta.At(q, j, tt+1))
				}
				beta.Set(math.Log(v), q, i, tt)
			}

			// t<nobs-1, entry states.
			var v float64
			for j := 1; j < nstates[q]-1; j++ {
				v += math.Exp(hmms[q].a.At(0, j) + hmms[q].logProb(j, obs[tt]) + beta.At(q, j, tt))
			}
			beta.Set(math.Log(v), q, 0, tt)
		}
	}
	return
}
