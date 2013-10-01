package hmm

import (
	"flag"
	"github.com/akualab/gjoa/model"
	"testing"
)

func TestViterbi(t *testing.T) {
	flag.Parse()
	hmm := MakeHMM(t)
	bt, logProbViterbi, err := hmm.viterbi(obs0)
	if err != nil {
		t.Fatal(err)
	}
	expectedViterbiLog := -26.8129904950932
	model.CompareFloats(t, expectedViterbiLog, logProbViterbi, "Error in logProbViterbi", epsilon)
	model.CompareSliceInt(t, viterbiSeq, bt, "Error in viterbi seq")
}

var (
	viterbiSeq = []int{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0}
)
