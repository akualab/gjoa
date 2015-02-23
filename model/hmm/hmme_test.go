package hmm

import (
	"math/rand"
	"os"
	"testing"

	"github.com/akualab/narray"
)

// Test embedded hmm.

/*
                     nq=2 (num models)
          q = 0                q = 1  (model index)
       N(q=0) = 5            N(q=1) = 4  (num states fro model q)

   0    1     2     3    4   0    1    2     3  (state indices)
   o-->( )-->( )-->( )-->o   o-->( )-->( )-->o
           |
        a(q=0, i=1, j=2)  (transition prob)

   Transition probs:

   q=0                   q=1
      0  1  2  3  4         0  1  2  3
   0     .9       .1     0     1
   1     .5 .5           1     .3 .2 .5
   2        .3 .6 .1     2        .6 .4
   3           .7 .3     3
   4

*/

const (
	nq = 2 // num models
)

var (
	obs               = []int{0, 0, 0, 1, 0, 1, 1, 1, 0, 1}
	a, b, alpha, beta *narray.NArray
	nstates           [nq]int
	ns, nobs          int
	r                 = rand.New(rand.NewSource(33))
)

func TestMain(m *testing.M) {

	ns = 5 // max num states in a model
	nstates[0] = 5
	nstates[1] = 4
	nobs = len(obs)
	a = narray.New(nq, ns, ns)
	b = narray.New(nq, ns, nobs)
	alpha = narray.New(nq, ns, nobs)
	beta = narray.New(nq, ns, nobs)

	a.Set(.9, 0, 0, 1)
	a.Set(.1, 0, 0, 4)
	a.Set(.5, 0, 1, 1)
	a.Set(.5, 0, 1, 2)
	a.Set(.3, 0, 2, 2)
	a.Set(.6, 0, 2, 3)
	a.Set(.1, 0, 2, 4)
	a.Set(.7, 0, 3, 3)
	a.Set(.3, 0, 3, 4)

	a.Set(1, 1, 0, 1)
	a.Set(.3, 1, 1, 1)
	a.Set(.2, 1, 1, 2)
	a.Set(.5, 1, 1, 3)
	a.Set(.6, 1, 2, 2)
	a.Set(.4, 1, 2, 3)

	someProbs := []float64{.2, .4, .5, .7} // make it easy to debug.
	for q := 0; q < nq; q++ {
		for i := 1; i < nstates[q]-1; i++ {
			for t := 0; t < nobs; t++ {
				k := r.Intn(len(someProbs))
				b.Set(someProbs[k], q, i, t)
			}
		}
	}

	os.Exit(m.Run())
}

func TestValues(t *testing.T) {

	t.Logf("num moldes: %d", nq)
	t.Logf("num observations: %d", nobs)

	for q := 0; q < nq; q++ {
		for i := 0; i < nstates[q]; i++ {
			for j := i; j < nstates[q]; j++ {
				p := a.At(q, i, j)
				if p > 0 {
					t.Logf("q:%d i:%d j:%d a:%.1f", q, i, j, p)
				}
			}
		}
	}

	for q := 0; q < nq; q++ {
		for i := 1; i < nstates[q]-1; i++ {
			for tt := 0; tt < nobs; tt++ {
				p := b.At(q, i, tt)
				if p > 0 {
					t.Logf("q:%d i:%d t:%d b:%.1f", q, i, tt, p)
				}

			}
		}
	}
}

func computeAlpha() {

	// t=0, entry state, first model.
	alpha.Set(1.0, 0, 0, 0)

	// t=0, entry state, after first model.
	for q := 1; q < nq; q++ {
		alpha.Set(alpha.At(q-1, 0, 0)*a.At(q-1, 0, nstates[q-1]-1), q, 0, 0)
	}

	// t=0, emitting states.
	for q := 0; q < nq; q++ {
		for j := 1; j < nstates[q]-1; j++ {
			v := a.At(q, 0, j) * b.At(q, j, 0)
			alpha.Set(v, q, j, 0)
		}
	}

	// t=0, exit states.
	for q := 0; q < nq; q++ {
		var v float64
		for i := 1; i < nstates[q]-1; i++ {
			v += alpha.At(q, i, 0) * a.At(q, i, nstates[q]-1)
		}
		alpha.Set(v, q, nstates[q]-1, 0)
	}

	// t>0, entry state, first model.
	// alpha(0,0,t) = 0

	// t>0, entry state, after first model.
	for q := 1; q < nq; q++ {
		for tt := 1; tt < nobs; tt++ {
			v := alpha.At(q-1, nstates[q-1]-1, tt-1) + alpha.At(q-1, 0, tt)*a.At(q-1, 0, nstates[q-1]-1)
			alpha.Set(v, q, 0, tt)
		}
	}

	// t>0, emitting states.
	for q := 0; q < nq; q++ {
		for j := 1; j < nstates[q]-1; j++ {
			for tt := 1; tt < nobs; tt++ {
				v := alpha.At(q, 0, tt) * a.At(q, 0, j)
				for i := 1; i < nstates[q]-1; i++ {
					v += alpha.At(q, i, tt-1) * a.At(q, i, j)
				}
				v *= b.At(q, j, tt)
				alpha.Set(v, q, j, tt)
			}
		}
	}

	// t>0, exit states.
	for q := 0; q < nq; q++ {
		for tt := 1; tt < nobs; tt++ {
			var v float64
			for i := 1; i < nstates[q]-1; i++ {
				v += alpha.At(q, i, tt) * a.At(q, i, nstates[q]-1)
			}
			alpha.Set(v, q, nstates[q]-1, tt)
		}
	}
}

func computeBeta() {

	// t=nobs-1, exit state, last model.
	beta.Set(1.0, nq-1, nstates[nq-1]-1, nobs-1)

	// t=nobs-1, exit state, before last model.
	for q := nq - 2; q >= 0; q-- {
		v := beta.At(q+1, nstates[q+1]-1, nobs-1) * a.At(q+1, 0, nstates[q+1]-1)
		beta.Set(v, q, nstates[q]-1, nobs-1)
	}

	// t=nobs-1, emitting states.
	for q := nq - 1; q >= 0; q-- {
		for i := nstates[q] - 2; i > 0; i-- {
			v := a.At(q, i, nstates[q]-1) * beta.At(q, nstates[q]-1, nobs-1)
			beta.Set(v, q, i, nobs-1)
		}
	}

	// t=nobs-1, entry states.
	for q := nq - 1; q >= 0; q-- {
		var v float64
		for j := 1; j < nstates[q]-1; j++ {
			v += a.At(q, 0, j) * b.At(q, j, nobs-1) * beta.At(q, j, nobs-1)
		}
		beta.Set(v, q, 0, nobs-1)
	}

	// t<nobs-1, exit state, last model.
	// beta(nq-1,nstates[nq-1]-1,t) = 0

	// t<nobs-1, exit state, before last model.
	for q := nq - 2; q >= 0; q-- {
		for tt := nobs - 2; tt >= 0; tt-- {
			v := beta.At(q+1, 0, tt+1) + beta.At(q+1, nstates[q+1]-1, tt)*a.At(q+1, 0, nstates[q+1]-1)
			beta.Set(v, q, nstates[q]-1, tt)
		}
	}

	// t<nobs-1, emitting states.
	for q := nq - 1; q >= 0; q-- {
		for tt := nobs - 2; tt >= 0; tt-- {
			for i := nstates[q] - 1; i >= 0; i-- {
				v := a.At(q, i, nstates[q]-1) * beta.At(q, nstates[q]-1, tt)
				for j := 1; j < nstates[q]-1; j++ {
					v += a.At(q, i, j) * b.At(q, j, tt+1) * beta.At(q, j, tt+1)
				}
				beta.Set(v, q, i, tt)
			}
		}
	}

	// t<nobs-1, entry states.
	for q := nq - 1; q >= 0; q-- {
		for tt := nobs - 2; tt >= 0; tt-- {
			var v float64
			for j := 1; j < nstates[q]-1; j++ {
				v += a.At(q, 0, j) * b.At(q, j, tt) * beta.At(q, j, tt)
			}
			beta.Set(v, q, 0, tt)
		}
	}
}

func TestAlphaBeta(t *testing.T) {

	computeAlpha()
	for q := 0; q < nq; q++ {
		for i := 0; i < nstates[q]; i++ {
			for tt := 0; tt < nobs; tt++ {
				p := alpha.At(q, i, tt)
				if p > epsilon {
					t.Logf("q:%d i:%d t:%d alpha:%12f", q, i, tt, p)
				}
			}
		}
	}

	computeBeta()
	for q := 0; q < nq; q++ {
		for i := 0; i < nstates[q]; i++ {
			for tt := 0; tt < nobs; tt++ {
				p := beta.At(q, i, tt)
				if p > epsilon {
					t.Logf("q:%d i:%d t:%d beta:%12f", q, i, tt, p)
				}
			}
		}
	}

	delta := alpha.At(nq-1, nstates[nq-1]-1, nobs-1) - beta.At(0, 0, 0)
	if delta > epsilon {
		t.Fatalf("alpha(nq-1,n[q-1]-1,nobs-1)=%f does not match beta(0,0,0)%f", alpha.At(nq-1, nstates[nq-1]-1, nobs-1), beta.At(0, 0, 0))
	}
}