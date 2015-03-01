package hmm

import (
	"math"
	"testing"

	"github.com/akualab/gjoa/model"
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

var (
	hmms2 *chain
	ms2   modelSet
)

func TestDebug(t *testing.T) {

	initChainFB2()
	t.Log("compute fb using package")
	hmms2.update()
	nq := hmms2.nq
	alpha2 := hmms2.alpha.At(nq-1, hmms2.ns[nq-1]-1, nobs-1)
	beta2 := hmms2.beta.At(0, 0, 0)

	t.Logf("alpha2:%f", alpha2)
	t.Logf("beta2:%f", beta2)

	// check log prob per obs calculated with alpha and beta
	delta := math.Abs(alpha2-beta2) / float64(nobs)
	if delta > 0.01 {
		t.Fatalf("alphaLogProb:%f does not match betaLogProb:%f", alpha2, beta2)
	}

	ms2.reestimate()

}

func initChainFB2() {

	hmm0 := newHMM("model 0", 0, narray.New(nstates[0], nstates[0]),
		[]model.Scorer{nil, newScorer(0, 1), newScorer(0, 2), newScorer(0, 3), nil})

	hmm1 := newHMM("model 1", 1, narray.New(nstates[1], nstates[1]),
		[]model.Scorer{nil, newScorer(1, 1), newScorer(1, 2), nil})

	hmm2 := newHMM("model 2", 2, narray.New(3, 3),
		[]model.Scorer{nil, scorer{[]float64{math.Log(0.4), math.Log(0.2), math.Log(0.4)}}, nil})

	hmm0.a.Set(.9, 0, 1)
	hmm0.a.Set(.1, 0, 4)
	hmm0.a.Set(.5, 1, 1)
	hmm0.a.Set(.5, 1, 2)
	hmm0.a.Set(.3, 2, 2)
	hmm0.a.Set(.6, 2, 3)
	hmm0.a.Set(.1, 2, 4)
	hmm0.a.Set(.7, 3, 3)
	hmm0.a.Set(.3, 3, 4)

	hmm1.a.Set(1, 0, 1)
	hmm1.a.Set(.3, 1, 1)
	hmm1.a.Set(.2, 1, 2)
	hmm1.a.Set(.5, 1, 3)
	hmm1.a.Set(.6, 2, 2)
	hmm1.a.Set(.4, 2, 3)

	hmm2.a.Set(1, 0, 1)
	hmm2.a.Set(0.5, 1, 1)
	hmm2.a.Set(0.5, 1, 2)

	hmm0.a = narray.Log(nil, hmm0.a.Copy())
	hmm1.a = narray.Log(nil, hmm1.a.Copy())
	hmm2.a = narray.Log(nil, hmm2.a.Copy())

	xobs := make([]model.Obs, nobs, nobs)
	for k, v := range obs {
		xobs[k] = model.NewIntObs(v, model.NoLabel())
	}

	ms2 = make(modelSet)
	//	hmms2 = newChain(ms2, xobs, hmm0, hmm1)
	hmms2 = newChain(ms2, xobs, hmm0, hmm1)
}
