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
