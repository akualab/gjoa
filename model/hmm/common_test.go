package hmm

import (
	"math"
	"math/rand"
	"testing"
)

func TestGetRandomStateFromDist(t *testing.T) {
	dist := []float64{0.5, 0.5}
	freq := []int{0, 0}
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 5000; i++ {
		ranint, err := GetRandomStateFromDist(dist, r)
		if err != nil {
			t.Fatalf("GetRandomStateFromDist failed")
		}
		freq[ranint] = freq[ranint] + 1
	}
	ratio := float64(freq[0]) / 5000.0
	if math.Abs(ratio-0.5) > 0.05 {
		t.Fatalf("GetRandomStateFromDist failed")
	}
}

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
		t.Fatalf("GetRandomVectoFromDist failed")
	}
	t.Logf("States %v", states)
	t.Logf("Seq %v", obs)
}
