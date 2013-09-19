package hmm

import (
	"testing"
)

func TestRandomVectorFromHMM(t *testing.T) {
	mean := []float64{1.0, 4.0}
	sd := []float64{1.0, 2.0}
	initialStateProbs := []float64{0.8, 0.2}
	transProbs := [][]float64{{0.9, 0.1}, {0.3, 0.7}}
	// size of the generated sequence
	n := 1000
	states, obs, err := GetRandomVectorFromHMM(
		transProbs, initialStateProbs, mean, sd, n)
	if err != nil {
		t.Fatalf("RandomVectoFromDist failed")
	}
	t.Logf("States %v", states)
	t.Logf("Seq %v", obs)
}
