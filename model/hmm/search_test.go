package hmm

import "github.com/akualab/graph"

type vvalue struct {
	f graph.ScoreFunc
}

// Implements the Viterbier interface.
func (v vvalue) ScoreFunction(n int, node *graph.Node) float64 {
	return v.f(n)
}

// func TestViterbi(t *testing.T) {
// 	flag.Parse()
// 	hmm := MakeHMM(t)
// 	bt, logProbViterbi, err := hmm.Viterbi(obs0)
// 	if err != nil {
// 		t.Fatal(err)
// 	}
// 	expectedViterbiLog := -26.8129904950932
// 	gjoa.CompareFloats(t, expectedViterbiLog, logProbViterbi, "Error in logProbViterbi", epsilon)
// 	gjoa.CompareSliceInt(t, viterbiSeq, bt, "Error in viterbi seq")
// }

// var (
// 	viterbiSeq = []int{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0}
// )

// func MakeGraph(t *testing.T) *graph.Graph {
// 	hmm := MakeHMM(t)
// 	// Define score functions to return state probabilities.
// 	var s1Func = func(n int, node *graph.Node) float64 {
// 		o := model.F64ToObs(obs0[n])
// 		return hmm.ObsModels[0].LogProb(o)
// 	}
// 	var s2Func = func(n int, node *graph.Node) float64 {
// 		o := model.F64ToObs(obs0[n])
// 		return hmm.ObsModels[1].LogProb(o)
// 	}
// 	var finalFunc = func(n int, node *graph.Node) float64 {
// 		return 0
// 	}
// 	g := graph.New()

// 	g.Set("s0", vvalue{}) // initial state
// 	g.Set("s1", vvalue{s1Func})
// 	g.Set("s2", vvalue{s2Func})
// 	g.Set("s3", vvalue{finalFunc}) // final state

// 	g.Connect("s0", "s1", 0.8)
// 	g.Connect("s0", "s2", 0.2)
// 	g.Connect("s1", "s1", 0.9)
// 	g.Connect("s1", "s2", 0.1)
// 	g.Connect("s2", "s1", 0.3)
// 	g.Connect("s2", "s2", 0.7)

// 	// Convert transition probabilities to log.
// 	g.ConvertToLogProbs()
// 	return g
// }

// func TestGraphViterbi(t *testing.T) {
// 	var start, end *graph.Node
// 	var e error

// 	// Create the graph.
// 	g := MakeGraph(t)
// 	fmt.Printf("simple graph:\n%s\n", g)
// 	//panic(e)

// 	// Define the start and end nodes.
// 	if start, e = g.Get("s0"); e != nil {
// 		panic(e)
// 	}
// 	if end, e = g.Get("s3"); e != nil {
// 		panic(e)
// 	}

// 	// Create a decoder.
// 	dec, e := graph.NewDecoder(g, start, end)
// 	if e != nil {
// 		panic(e)
// 	}

// 	// Find the optimnal sequence.
// 	token := dec.Decode(12)

// 	// The token has the backtrace to find the optimal path.
// 	fmt.Printf("\n\n>>>> FINAL: %s\n", token)
// 	fmt.Printf("Score: %f \n", token.Score)
// 	expectedViterbiLog := -26.8129904950932
// 	gjoa.CompareFloats(t, expectedViterbiLog, token.Score, "Error in TestGraphViterbi", epsilon)
// }
