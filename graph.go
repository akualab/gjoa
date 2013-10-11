package gjoa

import (
	"github.com/akualab/gjoa/floatx"
	"github.com/golang/glog"
	"github.com/gonum/floats"
	"io"
	"io/ioutil"
	"launchpad.net/goyaml"
	"os"
	"sort"
)

type Node struct {
	Name string `yaml:"name" json:"name"`
}

type Edge struct {
	FromName string  `yaml:"from" json:"from"`
	From     *Node   `yaml:"-" json:"-"`
	ToName   string  `yaml:"to" json:"to"`
	To       *Node   `yaml:"-" json:"-"`
	Weight   float64 `yaml:"weight" json:"weight"`
}

type Graph struct {
	Name  string  `yaml:"name" json:"name"`
	Edges []*Edge `yaml:"edges" json:"edges"`
	nodes map[string]*Node
}

// Reads graph from io.Reader and creates a new Graph instance.
func ReadGraph(r io.Reader) (*Graph, error) {

	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}

	// Read using Graph.
	g := &Graph{}
	g.nodes = make(map[string]*Node)
	err = goyaml.Unmarshal([]byte(b), g)
	if err != nil {
		return nil, err
	}
	g.createNodes()
	return g, nil
}

// Reads graph from file and creates a new Graph instance.
func ReadFile(fn string) (*Graph, error) {

	f, err := os.Open(fn)
	if err != nil {
		return nil, err
	}
	return ReadGraph(f)
}

// Writes Graph to an io.Writer.
func (g *Graph) Write(w io.Writer) error {

	b, err := goyaml.Marshal(g)
	if err != nil {
		return err
	}
	_, e := w.Write(b)
	return e
}

// Writes Graph to a file.
func (g *Graph) WriteFile(fn string) error {

	f, err := os.Create(fn)
	if err != nil {
		return err
	}
	defer f.Close()

	ee := g.Write(f)
	if ee != nil {
		return ee
	}

	return nil
}

// Returns the nodes in the graph and a transition probability
// matrix.
func (g *Graph) Nodes() (nodes []*Node, probs [][]float64) {

	n := len(g.nodes)
	probs = floatx.MakeFloat2D(n, n)

	// Put nodes in slice.
	nodes = make([]*Node, n)
	index := make(map[string]int)
	var k int
	for _, x := range g.nodes {
		nodes[k] = x
		k += 1
	}

	// Sort nodes by name.
	sort.Sort(ByName{nodes})

	// Map Node name to matrix index.
	for k, v := range nodes {
		index[v.Name] = k
	}

	// Put transition weights in matrix.
	for _, v := range g.Edges {
		i := index[v.FromName]
		j := index[v.ToName]
		probs[i][j] = v.Weight
	}

	// Make rows add to 1.
	for i, v := range probs {
		sum := floats.Sum(probs[i])
		for j, _ := range v {
			probs[i][j] = probs[i][j] / sum
		}
	}
	return
}

// Create the nodes given the names.
func (g *Graph) createNodes() error {

	for _, v := range g.Edges {
		from := v.FromName
		to := v.ToName

		// Creates Node if it doesn't exist.
		if _, ok := g.nodes[from]; !ok {
			g.nodes[from] = &Node{Name: from}
		}
		if _, ok := g.nodes[to]; !ok {
			g.nodes[to] = &Node{Name: to}
		}

		// Add Node object to Edge.
		v.From = g.nodes[from]
		v.To = g.nodes[to]
	}

	glog.Infof("Read %d nodes.", len(g.nodes))
	return nil
}

// Sort Nodes.

type Nodes []*Node

func (s Nodes) Len() int      { return len(s) }
func (s Nodes) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

// ByName implements sort.Interface by providing Less and using the Len and
type ByName struct{ Nodes }

func (s ByName) Less(i, j int) bool { return s.Nodes[i].Name < s.Nodes[j].Name }
