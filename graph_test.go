package gjoa

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestGraph(t *testing.T) {

	// Create graph yaml file.
	fn := os.TempDir() + "graph.yaml"
	err := ioutil.WriteFile(fn, []byte(graphData), 0644)
	if err != nil {
		t.Fatal(err)
	}

	g, e := ReadFile(fn)
	if e != nil {
		t.Fatal(e)
	}

	t.Logf("Graph:\n,%+v", *g)
	t.Logf("Edges[0]:\n,%+v", *g.Edges[0])

	fn = os.TempDir() + "graph-out.yaml"
	e = g.WriteFile(fn)
	if e != nil {
		t.Fatal(e)
	}
	t.Logf("Wrote to file %s", fn)

	// Get transition probs.
	nodes, tpm := g.NodesAndProbs()
	for k, v := range tpm {
		t.Logf("From: %20s: %v", nodes[k].Name, v)
	}

	// Check values.
	CompareSliceFloat(t, expectedProbs[0], tpm[0], "Error in row 0", 0.0001)
	CompareSliceFloat(t, expectedProbs[1], tpm[1], "Error in row 1", 0.0001)
	CompareSliceFloat(t, expectedProbs[2], tpm[2], "Error in row 2", 0.0001)
	CompareSliceFloat(t, expectedProbs[3], tpm[3], "Error in row 3", 0.0001)

	if len(tpm[5]) > 0 {
		t.Fatalf("Expected nil, got %v.", tpm[5])
	}
}

const graphData string = `
name: southcourt
edges:
  - {from: BACKYARD, to: DINING, weight: 2.0}
  - {from: BACKYARD, to: LIVING, weight: 1.0}
  - {from: BACKYARD, to: KITCHEN, weight: 1.0}
  - {from: BATH1, to: BED1, weight: 3.0}
  - {from: BATH1, to: BED2, weight: 2.0}
  - {from: BATH2, to: BED4, weight: 2.0}
  - {from: BATH2, to: BED2, weight: 2.0}
  - {from: BATH3, to: BED5, weight: 2.0}
  - {from: BATH3, to: DINING, weight: 3.0}
  - {from: BED1, to: BED4, weight: 2.0}
  - {from: BED1, to: BATH1, weight: 2.0}
`

var expectedProbs = [][]float64{
	{0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.25, 0.25},
	{0, 0, 0, 0, 0.6, 0.4, 0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0},
	{0, 0, 0, 0, 0, 0, 0, 0.4, 0.6, 0, 0},
}
