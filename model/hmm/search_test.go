package hmm

import (
	"flag"
	"github.com/akualab/gjoa"
	"testing"
)

func TestViterbi(t *testing.T) {
	flag.Parse()
	hmm := MakeHMM(t)
	bt, logProbViterbi, err := hmm.Viterbi(obs0)
	if err != nil {
		t.Fatal(err)
	}
	expectedViterbiLog := -26.8129904950932
	gjoa.CompareFloats(t, expectedViterbiLog, logProbViterbi, "Error in logProbViterbi", epsilon)
	gjoa.CompareSliceInt(t, viterbiSeq, bt, "Error in viterbi seq")
}

var (
	viterbiSeq = []int{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0}
)
