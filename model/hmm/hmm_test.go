package hmm

import (
	"flag"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/akualab/gjoa/model"
	"github.com/akualab/narray"
	"github.com/golang/glog"
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
   0     1               0     1
   1     .5 .5           1     .3 .2 .5
   2        .3 .6 .1     2        .6 .4
   3           .7 .3     3
   4

*/

const (
	nq           = 2 // num models
	small        = 0.000001
	expectedProb = -33.575758
)

var (
	obs                           = []int{2, 0, 0, 1, 0, 2, 1, 1, 0, 1, 1, 1, 2, 1, 2, 0, 0, 1, 2, 1}
	xobs                          model.Obs
	nsymb                         = 3 // num distinct observations: 0,1,2
	a, b, loga, logb, alpha, beta *narray.NArray
	nstates                       [nq]int
	ns, nobs                      int
	r                             = rand.New(rand.NewSource(33))
	outputProbs                   *narray.NArray
	nchain                        Chain
	hmms                          *chain
	hmm0, hmm1                    *Net
	ms                            *Set
)

func TestMain(m *testing.M) {

	// configure glog.
	flag.Set("alsologtostderr", "true")
	flag.Set("log_dir", "/tmp/log")
	flag.Set("v", "3")
	flag.Parse()
	glog.Info("Logging configured")

	ns = 5 // max num states in a model
	nstates[0] = 5
	nstates[1] = 4
	nobs = len(obs)
	a = narray.New(nq, ns, ns)
	b = narray.New(nq, ns, nobs)
	alpha = narray.New(nq, ns, nobs)
	beta = narray.New(nq, ns, nobs)

	a.Set(1, 0, 0, 1)
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

	dist := [][]float64{{.4, .5, .1}, {.3, .5, .2}} // prob dist for states in model 0 and 1

	// output probs as a function of model,state,time
	for q := 0; q < nq; q++ {
		for i := 1; i < nstates[q]-1; i++ {
			for t := 0; t < nobs; t++ {
				p := dist[q][obs[t]]
				//				k := r.Intn(len(someProbs))
				//				b.Set(someProbs[k], q, i, t)
				b.Set(p, q, i, t)
			}
		}
	}

	// same output probs but as a function of model,state,symbol
	// we need this to test the network implementation.
	outputProbs = narray.New(nq, ns, nsymb)
	for q := 0; q < nq; q++ {
		for i := 1; i < nstates[q]-1; i++ {
			for k := 0; k < nsymb; k++ {
				p := math.Log(dist[q][k])
				outputProbs.Set(p, q, i, k)
			}
		}
	}

	loga = narray.Log(loga, a.Copy())
	logb = narray.Log(logb, b.Copy())

	data := make([][]float64, len(obs), len(obs))
	for k, v := range obs {
		data[k] = []float64{float64(v)}
	}
	xobs = model.NewFloatObsSequence(data, model.SimpleLabel(""), "")

	os.Exit(m.Run())
}

func TestValues(t *testing.T) {

	t.Logf("num moldes: %d", nq)
	t.Logf("num observations: %d", nobs)

	for q := 0; q < nq; q++ {
		for i := 0; i < nstates[q]; i++ {
			for j := i; j < nstates[q]; j++ {
				t.Logf("q:%d i:%d j:%d a:%.1f", q, i, j, a.At(q, i, j))
				t.Logf("q:%d i:%d j:%d loga:%.1f", q, i, j, loga.At(q, i, j))
			}
		}
	}

	for q := 0; q < nq; q++ {
		for i := 1; i < nstates[q]-1; i++ {
			for tt := 0; tt < nobs; tt++ {
				t.Logf("q:%d i:%d t:%d b:%.1f", q, i, tt, b.At(q, i, tt))
				t.Logf("q:%d i:%d t:%d logb:%.1f", q, i, tt, logb.At(q, i, tt))
			}
		}
	}
}

func computeAlpha(t *testing.T) {

	// t=0, entry state, first model.
	alpha.Set(1.0, 0, 0, 0)
	printLog(t, "alpha", 0, 0, 0, alpha)

	// t=0, entry state, after first model.
	for q := 1; q < nq; q++ {
		alpha.Set(alpha.At(q-1, 0, 0)*a.At(q-1, 0, nstates[q-1]-1), q, 0, 0)
		printLog(t, "alpha", q, 0, 0, alpha)
	}

	// t=0, emitting states.
	for q := 0; q < nq; q++ {
		for j := 1; j < nstates[q]-1; j++ {
			v := alpha.At(q, 0, 0) * a.At(q, 0, j) * b.At(q, j, 0)
			alpha.Set(v, q, j, 0)
			printLog(t, "alpha", q, j, 0, alpha)
		}
	}

	// t=0, exit states.
	for q := 0; q < nq; q++ {
		var v float64
		for i := 1; i < nstates[q]-1; i++ {
			v += alpha.At(q, i, 0) * a.At(q, i, nstates[q]-1)
		}
		alpha.Set(v, q, nstates[q]-1, 0)
		printLog(t, "alpha", q, nstates[q]-1, 0, alpha)
	}

	for q := 0; q < nq; q++ {
		for tt := 1; tt < nobs; tt++ {
			if q == 0 {
				// t>0, entry state, first model.
				// alpha(0,0,t) = 0
				printLog(t, "alpha", 0, 0, tt, alpha)
			} else {
				// t>0, entry state, after first model.
				v := alpha.At(q-1, nstates[q-1]-1, tt-1) + alpha.At(q-1, 0, tt)*a.At(q-1, 0, nstates[q-1]-1)
				alpha.Set(v, q, 0, tt)
				printLog(t, "alpha", q, 0, tt, alpha)
			}

			// t>0, emitting states.

			for j := 1; j < nstates[q]-1; j++ {
				v := alpha.At(q, 0, tt) * a.At(q, 0, j)
				for i := 1; i < nstates[q]-1; i++ {
					v += alpha.At(q, i, tt-1) * a.At(q, i, j)
				}
				v *= b.At(q, j, tt)
				alpha.Set(v, q, j, tt)
				printLog(t, "alpha", q, j, tt, alpha)
			}

			// t>0, exit states.
			var v float64
			for i := 1; i < nstates[q]-1; i++ {
				v += alpha.At(q, i, tt) * a.At(q, i, nstates[q]-1)
			}
			alpha.Set(v, q, nstates[q]-1, tt)
			printLog(t, "alpha", q, nstates[q]-1, tt, alpha)
		}
	}
}

func computeLogAlpha(t *testing.T) {

	alpha.SetValue(math.Inf(-1))

	// t=0, entry state, first model.
	alpha.Set(0.0, 0, 0, 0)
	printLog(t, "log_alpha", 0, 0, 0, alpha)

	// t=0, entry state, after first model.
	for q := 1; q < nq; q++ {
		v := alpha.At(q-1, 0, 0) + loga.At(q-1, 0, nstates[q-1]-1)
		alpha.Set(v, q, 0, 0)
		printLog(t, "log_alpha", q, 0, 0, alpha)
	}

	// t=0, emitting states.
	for q := 0; q < nq; q++ {
		for j := 1; j < nstates[q]-1; j++ {
			v := alpha.At(q, 0, 0) + loga.At(q, 0, j) + logb.At(q, j, 0)
			alpha.Set(v, q, j, 0)
			printLog(t, "log_alpha", q, j, 0, alpha)
		}
	}

	// t=0, exit states.
	for q := 0; q < nq; q++ {
		var v float64
		for i := 1; i < nstates[q]-1; i++ {
			v += math.Exp(alpha.At(q, i, 0) + loga.At(q, i, nstates[q]-1))
		}
		alpha.Set(math.Log(v), q, nstates[q]-1, 0)
		printLog(t, "log_alpha", q, nstates[q]-1, 0, alpha)
	}

	for q := 0; q < nq; q++ {
		for tt := 1; tt < nobs; tt++ {
			if q == 0 {
				// t>0, entry state, first model.
				// alpha(0,0,t) = 0
			} else {
				// t>0, entry state, after first model.
				v := math.Exp(alpha.At(q-1, nstates[q-1]-1, tt-1)) + math.Exp(alpha.At(q-1, 0, tt)+loga.At(q-1, 0, nstates[q-1]-1))
				alpha.Set(math.Log(v), q, 0, tt)
				printLog(t, "log_alpha", q, 0, tt, alpha)
			}

			// t>0, emitting states.
			for j := 1; j < nstates[q]-1; j++ {
				v := math.Exp(alpha.At(q, 0, tt) + loga.At(q, 0, j))
				for i := 1; i < nstates[q]-1; i++ {
					v += math.Exp(alpha.At(q, i, tt-1) + loga.At(q, i, j))
				}
				v = math.Log(v) + logb.At(q, j, tt)
				alpha.Set(v, q, j, tt)
				printLog(t, "log_alpha", q, j, tt, alpha)
			}

			// t>0, exit states.
			var v float64
			for i := 1; i < nstates[q]-1; i++ {
				v += math.Exp(alpha.At(q, i, tt) + loga.At(q, i, nstates[q]-1))
			}
			alpha.Set(math.Log(v), q, nstates[q]-1, tt)
			printLog(t, "log_alpha", q, nstates[q]-1, tt, alpha)
		}
	}
}

func computeBeta(t *testing.T) {

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

	for q := nq - 1; q >= 0; q-- {
		for tt := nobs - 2; tt >= 0; tt-- {

			if q == nq-1 {
				// t<nobs-1, exit state, last model.
				// beta(nq-1,nstates[nq-1]-1,t) = 0
			} else {
				// t<nobs-1, exit state, before last model.
				v := beta.At(q+1, 0, tt+1) + beta.At(q+1, nstates[q+1]-1, tt)*a.At(q+1, 0, nstates[q+1]-1)
				beta.Set(v, q, nstates[q]-1, tt)
			}

			// t<nobs-1, emitting states.
			for i := nstates[q] - 2; i > 0; i-- {
				v := a.At(q, i, nstates[q]-1) * beta.At(q, nstates[q]-1, tt)
				for j := 1; j < nstates[q]-1; j++ {
					v += a.At(q, i, j) * b.At(q, j, tt+1) * beta.At(q, j, tt+1)
				}
				beta.Set(v, q, i, tt)
			}

			// t<nobs-1, entry states.
			var v float64
			for j := 1; j < nstates[q]-1; j++ {
				v += a.At(q, 0, j) * b.At(q, j, tt) * beta.At(q, j, tt)
			}
			beta.Set(v, q, 0, tt)
		}
	}
}

func computeLogBeta(t *testing.T) {

	beta.SetValue(math.Inf(-1))

	// t=nobs-1, exit state, last model.
	beta.Set(0, nq-1, nstates[nq-1]-1, nobs-1)
	printLog(t, "log_beta", nq-1, nstates[nq-1]-1, nobs-1, beta)

	// t=nobs-1, exit state, before last model.
	for q := nq - 2; q >= 0; q-- {
		v := beta.At(q+1, nstates[q+1]-1, nobs-1) + loga.At(q+1, 0, nstates[q+1]-1)
		beta.Set(v, q, nstates[q]-1, nobs-1)
		printLog(t, "log_beta", q, nstates[q]-1, nobs-1, beta)
	}

	// t=nobs-1, emitting states.
	for q := nq - 1; q >= 0; q-- {
		for i := nstates[q] - 2; i > 0; i-- {
			v := loga.At(q, i, nstates[q]-1) + beta.At(q, nstates[q]-1, nobs-1)
			beta.Set(v, q, i, nobs-1)
			printLog(t, "log_beta", q, i, nobs-1, beta)
		}
	}

	// t=nobs-1, entry states.
	for q := nq - 1; q >= 0; q-- {
		var v float64
		for j := 1; j < nstates[q]-1; j++ {
			v += math.Exp(loga.At(q, 0, j) + logb.At(q, j, nobs-1) + beta.At(q, j, nobs-1))
		}
		beta.Set(math.Log(v), q, 0, nobs-1)
		printLog(t, "log_beta", q, 0, nobs-1, beta)
	}

	for q := nq - 1; q >= 0; q-- {
		for tt := nobs - 2; tt >= 0; tt-- {

			if q == nq-1 {
				// t<nobs-1, exit state, last model.
				//	for tt := nobs - 2; tt >= 0; tt-- {
				//		beta.Set(-math.MaxFloat64, nq-1, nstates[nq-1]-1, tt)
				//	}
				printLog(t, "log_beta", q, nstates[nq-1]-1, tt, beta)
			} else {
				// t<nobs-1, exit state, before last model.
				v := math.Exp(beta.At(q+1, 0, tt+1)) + math.Exp(beta.At(q+1, nstates[q+1]-1, tt)+loga.At(q+1, 0, nstates[q+1]-1))
				beta.Set(math.Log(v), q, nstates[q]-1, tt)
				printLog(t, "log_beta", q, nstates[q]-1, tt, beta)
			}
			// t<nobs-1, emitting states.
			for i := nstates[q] - 2; i > 0; i-- {
				v := math.Exp(loga.At(q, i, nstates[q]-1) + beta.At(q, nstates[q]-1, tt))
				for j := 1; j < nstates[q]-1; j++ {
					v += math.Exp(loga.At(q, i, j) + logb.At(q, j, tt+1) + beta.At(q, j, tt+1))
				}
				beta.Set(math.Log(v), q, i, tt)
				printLog(t, "log_beta", q, i, tt, beta)
			}

			// t<nobs-1, entry states.
			var v float64
			for j := 1; j < nstates[q]-1; j++ {
				v += math.Exp(loga.At(q, 0, j) + logb.At(q, j, tt) + beta.At(q, j, tt))
			}
			beta.Set(math.Log(v), q, 0, tt)
			printLog(t, "log_beta", q, 0, tt, beta)
		}
	}
}

func TestAlphaBeta(t *testing.T) {

	initChainFB(t)
	var err error
	hmms, err = ms.chainFromNets(xobs, hmm0, hmm1)
	if err != nil {
		t.Fatal(err)
	}

	computeAlpha(t)
	computeBeta(t)
	alpha1 := alpha.At(nq-1, nstates[nq-1]-1, nobs-1)
	beta1 := beta.At(0, 0, 0)
	delta := math.Abs(alpha1 - beta1)
	if delta > small {
		t.Fatalf("alpha(nq-1,n[q-1]-1,nobs-1)=%e does not match beta(0,0,0)%e", alpha.At(nq-1, nstates[nq-1]-1, nobs-1), beta.At(0, 0, 0))
	}

	t.Logf("alpha:%e beta:%e", alpha.At(nq-1, nstates[nq-1]-1, nobs-1), beta.At(0, 0, 0))
	computeLogAlpha(t)
	computeLogBeta(t)
	t.Logf("log_alpha:%f log_beta:%f", alpha.At(nq-1, nstates[nq-1]-1, nobs-1), beta.At(0, 0, 0))
	t.Logf("alpha1:%f alpha2:%f", math.Log(alpha1), alpha.At(nq-1, nstates[nq-1]-1, nobs-1))
	t.Logf("beta1:%f beta2:%f", math.Log(beta1), beta.At(0, 0, 0))

	alpha1 = alpha.At(nq-1, nstates[nq-1]-1, nobs-1)
	beta1 = beta.At(0, 0, 0)

	t.Log("")
	t.Log("compute fb using package")
	hmms.update()
	alpha2 := hmms.alpha.At(nq-1, nstates[nq-1]-1, nobs-1)
	beta2 := hmms.beta.At(0, 0, 0)

	t.Logf("alpha1:%f alpha2:%f", alpha1, alpha2)
	t.Logf("beta1:%f beta2:%f", beta1, beta2)

	delta = math.Abs(alpha1 - alpha2)
	if delta > smallNumber {
		t.Fatalf("log_alpha:%f does not match x_alpha:%f", alpha1, alpha2)
	}

	delta = math.Abs(beta1 - beta2)
	if delta > smallNumber {
		t.Fatalf("log_beta:%f does not match x_alpha:%f", beta1, beta2)
	}

	// check log prob per obs calculated with alpha and beta
	delta = math.Abs(alpha2-beta2) / float64(nobs)
	if delta > 0.01 {
		t.Fatalf("alphaLogProb:%e does not match betaLogProb:%f", alpha2, beta2)
	}

	ms.reestimate()

}

type scorer struct {
	testModel
	op []float64
}

func newScorer(model, state int) scorer {
	sc := scorer{op: make([]float64, nsymb, nsymb)}
	for k := 0; k < nsymb; k++ {
		sc.op[k] = outputProbs.At(model, state, k)
	}
	return sc
}

func (s scorer) LogProb(o model.Obs) float64 {
	return s.op[int(o.Value().([]float64)[0])]
}

func printLog(t *testing.T, name string, q, i, tt int, a *narray.NArray) {
	t.Logf("q:%d i:%d t:%d %s:%12f", q, i, tt, name, a.At(q, i, tt))
}

func (m *Net) testLogProb(s, o int) float64 {
	return m.B[s].LogProb(model.NewIntObs(o, model.SimpleLabel(""), ""))
}

func initChainFB(t *testing.T) {

	var err error
	h0 := narray.New(nstates[0], nstates[0])
	h1 := narray.New(nstates[1], nstates[1])

	h0.Set(1, 0, 1)
	h0.Set(.5, 1, 1)
	h0.Set(.5, 1, 2)
	h0.Set(.3, 2, 2)
	h0.Set(.6, 2, 3)
	h0.Set(.1, 2, 4)
	h0.Set(.7, 3, 3)
	h0.Set(.3, 3, 4)

	h1.Set(1, 0, 1)
	h1.Set(.3, 1, 1)
	h1.Set(.2, 1, 2)
	h1.Set(.5, 1, 3)
	h1.Set(.6, 2, 2)
	h1.Set(.4, 2, 3)

	h0 = narray.Log(nil, h0.Copy())
	h1 = narray.Log(nil, h1.Copy())

	ms, _ = NewSet()
	hmm0, err = ms.NewNet("model 0", h0,
		[]model.Modeler{nil, newScorer(0, 1), newScorer(0, 2), newScorer(0, 3), nil})
	fatalIf(t, err)
	hmm1, err = ms.NewNet("model 1", h1,
		[]model.Modeler{nil, newScorer(1, 1), newScorer(1, 2), nil})
	fatalIf(t, err)
}

func TestTeeModel(t *testing.T) {

	initChainFB(t)
	ms2, e := NewSet(hmm0, hmm1)
	fatalIf(t, e)
	testScorer := func() scorer {
		return scorer{op: []float64{math.Log(0.4), math.Log(0.2), math.Log(0.4)}}
	}

	h2 := narray.New(3, 3)
	h3 := narray.New(4, 4)

	h2.Set(1, 0, 1)
	h2.Set(0.5, 1, 1)
	h2.Set(0.5, 1, 2)

	h3.Set(.9, 0, 1)
	h3.Set(.1, 0, 3) // entry to exit transition.
	h3.Set(0.5, 1, 1)
	h3.Set(0.5, 1, 2)
	h3.Set(0.5, 2, 2)
	h3.Set(0.5, 2, 3)

	h2 = narray.Log(nil, h2.Copy())
	h3 = narray.Log(nil, h3.Copy())

	hmm2, err := ms2.NewNet("model 2", h2,
		[]model.Modeler{nil, testScorer(), nil})
	fatalIf(t, err)

	hmm3, errr := ms2.NewNet("model 3", h3,
		[]model.Modeler{nil, testScorer(), testScorer(), nil})
	fatalIf(t, errr)

	hmms2, err := ms2.chainFromNets(xobs, hmm0, hmm2, hmm3, hmm0, hmm0, hmm0, hmm0, hmm2)
	if err != nil {
		t.Fatal(err)
	}

	hmms2.update()
	nq := hmms2.nq
	alpha2 := hmms2.alpha.At(nq-1, hmms2.ns[nq-1]-1, nobs-1)
	beta2 := hmms2.beta.At(0, 0, 0)

	t.Logf("alpha2:%f", alpha2)
	t.Logf("beta2:%f", beta2)

	// check log prob per obs calculated with alpha and beta
	delta := math.Abs(alpha2-beta2) / float64(nobs)
	if delta > smallNumber {
		t.Fatalf("alphaLogProb:%f does not match betaLogProb:%f", alpha2, beta2)
	}

	_, err = ms2.chainFromNets(xobs, hmm3, hmm0, hmm2, hmm2)
	if err == nil {
		t.Fatal("expected error, got nil - first model has trans from entry to exit")
	}

	_, err = ms2.chainFromNets(xobs, hmm1, hmm0, hmm2, hmm3)
	if err == nil {
		t.Errorf("expected error, got nil - last model has trans from entry to exit")
	}
}

func TestLR(t *testing.T) {

	initChainFB(t)
	ms2, e := NewSet(hmm0, hmm1)
	fatalIf(t, e)
	testScorer := func() scorer {
		return scorer{op: []float64{math.Log(0.4), math.Log(0.2), math.Log(0.4)}}
	}

	hmm2, err := ms2.makeLetfToRight("model 2", 4, 0.4, 0.1,
		[]model.Modeler{nil, testScorer(), testScorer(), nil})
	fatalIf(t, err)
	hmm3, errr := ms2.makeLetfToRight("model 3", 6, 0.3, 0.2,
		[]model.Modeler{nil, testScorer(), testScorer(), testScorer(), testScorer(), nil})
	fatalIf(t, errr)
	hmms2, err := ms2.chainFromNets(xobs, hmm0, hmm2, hmm3, hmm0, hmm0, hmm0, hmm0, hmm2)
	if err != nil {
		t.Fatal(err)
	}

	hmms2.update()
	nq := hmms2.nq
	alpha2 := hmms2.alpha.At(nq-1, hmms2.ns[nq-1]-1, nobs-1)
	beta2 := hmms2.beta.At(0, 0, 0)

	t.Logf("alpha2:%f", alpha2)
	t.Logf("beta2:%f", beta2)

	// check log prob per obs calculated with alpha and beta
	delta := math.Abs(alpha2-beta2) / float64(nobs)
	if delta > smallNumber {
		t.Fatalf("alphaLogProb:%f does not match betaLogProb:%f", alpha2, beta2)
	}
	if math.Abs(alpha2-expectedProb) > smallNumber {
		t.Fatalf("alphaLogProb:%f, expected:%f", alpha2, expectedProb)
	}

}

func TestLRAssign(t *testing.T) {

	initChainFB(t)
	ms2, e := NewSet(hmm0, hmm1)
	fatalIf(t, e)
	testScorer := func() scorer {
		return scorer{op: []float64{math.Log(0.4), math.Log(0.2), math.Log(0.4)}}
	}

	_, err := ms2.makeLetfToRight("model 2", 4, 0.4, 0.1,
		[]model.Modeler{nil, testScorer(), testScorer(), nil})
	fatalIf(t, err)
	_, errr := ms2.makeLetfToRight("model 3", 6, 0.3, 0.2,
		[]model.Modeler{nil, testScorer(), testScorer(), testScorer(), testScorer(), nil})
	fatalIf(t, errr)

	// we need to put a label to use assigner - use same sequence as in TestLR() using the model names
	simplelab := model.SimpleLabel("model 0,model 2,model 3,model 0,model 0,model 0,model 0,model 2")
	oo := model.NewFloatObsSequence(xobs.Value().([][]float64), simplelab, "")

	// read it back
	sl := oo.Label().(model.SimpleLabel)
	glog.V(5).Infoln("read label: ", sl)
	var assigner DirectAssigner
	hmms2, err := ms2.chainFromAssigner(oo, assigner)
	if err != nil {
		t.Fatal(err)
	}

	hmms2.update()
	nq := hmms2.nq
	alpha2 := hmms2.alpha.At(nq-1, hmms2.ns[nq-1]-1, nobs-1)
	beta2 := hmms2.beta.At(0, 0, 0)

	t.Logf("alpha2:%f", alpha2)
	t.Logf("beta2:%f", beta2)

	// check log prob per obs calculated with alpha and beta
	delta := math.Abs(alpha2-beta2) / float64(nobs)
	if delta > smallNumber {
		t.Fatalf("alphaLogProb:%f does not match betaLogProb:%f", alpha2, beta2)
	}
	if math.Abs(alpha2-expectedProb) > smallNumber {
		t.Fatalf("alphaLogProb:%f, expected:%f", alpha2, expectedProb)
	}
}

func TestHMMModel(t *testing.T) {

	initChainFB(t)
	modelSet, e := NewSet(hmm0, hmm1)
	fatalIf(t, e)
	testScorer := func() scorer {
		return scorer{op: []float64{math.Log(0.4), math.Log(0.2), math.Log(0.4)}}
	}

	_, err := modelSet.makeLetfToRight("model 2", 4, 0.4, 0.1,
		[]model.Modeler{nil, testScorer(), testScorer(), nil})
	fatalIf(t, err)
	_, errr := modelSet.makeLetfToRight("model 3", 6, 0.3, 0.2,
		[]model.Modeler{nil, testScorer(), testScorer(), testScorer(), testScorer(), nil})
	fatalIf(t, errr)

	// we need to put a label to use assigner - use same sequence as in TestLR() using the model names
	simplelab := model.SimpleLabel("model 0,model 2,model 3,model 0,model 0,model 0,model 0,model 2")
	oo := model.NewFloatObsSequence(xobs.Value().([][]float64), simplelab, "")

	// read it back
	var assigner DirectAssigner
	hmm := NewModel(OSet(modelSet), OAssign(assigner))
	hmm.UpdateOne(oo, model.NoWeight(oo))
	hmm.Estimate()

	hmm.Clear()
	hmm.UpdateOne(oo, model.NoWeight(oo))
	hmm.Estimate()

	hmm.Clear()
	hmm.UpdateOne(oo, model.NoWeight(oo))
	hmm.Estimate()

	hmm.Clear()
	hmm.UpdateOne(oo, model.NoWeight(oo))
	hmm.Estimate()
}

func fatalIf(t *testing.T, err error) {

	if err != nil {
		t.Fatal(err)
	}
}

func TestModelName(t *testing.T) {

	ms2, _ := NewSet()
	_, err := ms2.NewNet("XXX", narray.New(3, 3),
		[]model.Modeler{nil, nil, nil})
	fatalIf(t, err)

	_, errr := ms2.NewNet("XXX", narray.New(4, 4),
		[]model.Modeler{nil, nil, nil, nil})

	if errr == nil {
		t.Fatalf("expected error, got nil for duplicate model name")
	}
}

type testModel struct{}

func (m testModel) Name() string                                             { return "" }
func (m testModel) Dim() int                                                 { return 0 }
func (m testModel) Update(x model.Observer, w func(model.Obs) float64) error { return nil }
func (m testModel) UpdateOne(o model.Obs, w float64)                         {}
func (m testModel) Estimate() error                                          { return nil }
func (m testModel) Clear()                                                   {}
func (m testModel) Predict(x model.Observer) ([]model.Labeler, error)        { return nil, nil }
func (m testModel) LogProb(x model.Obs) float64                              { return 0 }
func (m testModel) Sample() model.Obs                                        { return nil }
func (m testModel) SampleChan(size int) <-chan model.Obs                     { return nil }
