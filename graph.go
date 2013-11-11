package gjoa

import (
	"bytes"
	"fmt"
	"github.com/golang/glog"
	"github.com/gonum/floats"
	"io"
	"io/ioutil"
	"launchpad.net/goyaml"
	"os"
	"sort"
)

type Node struct {
	Name string                 `yaml:"name" json:"name"`
	Atts map[string]interface{} `yaml:"atts" json:"atts"`
}

type Edge struct {
	FromName string  `yaml:"from" json:"from"`
	From     *Node   `yaml:"-" json:"-"`
	ToName   string  `yaml:"to" json:"to"`
	To       *Node   `yaml:"-" json:"-"`
	Weight   float64 `yaml:"weight" json:"weight"`
}

type Graph struct {
	Name  string           `yaml:"name" json:"name"`
	Edges []*Edge          `yaml:"edges,flow" json:"edges"`
	Nodes map[string]*Node `yaml:"nodes,flow" json:"nodes"`
}

// Reads graph from io.Reader and creates a new Graph instance.
func ReadGraph(r io.Reader) (*Graph, error) {

	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}

	// Read using Graph.
	g := &Graph{}
	g.Nodes = make(map[string]*Node)
	err = goyaml.Unmarshal([]byte(b), g)
	if err != nil {
		return nil, err
	}
	if glog.V(3) {
		glog.Infof("Graph Name: %s", g.Name)
		for k, v := range g.Edges {
			glog.Infof("Edge %2d:%+v", k, *v)
		}
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
	defer f.Close()
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

	if len(fn) == 0 {
		glog.Fatalf("Missing filename.")
	}

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

// Returns the nodes and transition probability
// matrix. The matrix is a [][]float64. Note that
// rows with no outgoing edges  have a nil slice.
func (g *Graph) NodesAndProbs() (nodes []*Node, probs [][]float64) {

	n := len(g.Nodes)
	probs = make([][]float64, n)

	// Put nodes in slice.
	nodes = make([]*Node, n)
	index := make(map[string]int)
	var k int
	for _, x := range g.Nodes {
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
		if len(probs[i]) == 0 {
			probs[i] = make([]float64, n)
		}
		probs[i][j] = v.Weight
	}

	// Make rows add to 1.
	for i, v := range probs {
		if len(probs[i]) == 0 {
			continue
		}
		sum := floats.Sum(probs[i])
		if sum == 0.0 {
			glog.Infof("WARNING: sum of graph weights for row %d is zero. (Hint: use floats in yaml file.)", i)
			continue
		}
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
		if _, ok := g.Nodes[from]; !ok {
			g.Nodes[from] = &Node{Name: from}
		}
		if _, ok := g.Nodes[to]; !ok {
			g.Nodes[to] = &Node{Name: to}
		}

		// Add Node object to Edge.
		v.From = g.Nodes[from]
		v.To = g.Nodes[to]
	}

	glog.Infof("Created %d nodes.", len(g.Nodes))
	return nil
}

func (g *Graph) appendEdge(from, to string, w float64) {
	g.Edges = append(g.Edges, &Edge{FromName: from, ToName: to, Weight: w})
}

// Inserts a node between each pair of connected nodes. Assigns a weight of 1.0 to
// the self-transition and the transition from the new node to the next node.
func (g *Graph) InsertContextDependentStates() (ng *Graph, cdNodes map[string]bool) {

	cdNodes = make(map[string]bool)
	ng = &Graph{Name: g.Name + " CD"}
	ng.Nodes = make(map[string]*Node)
	ng.Edges = make([]*Edge, 0, 3*len(g.Edges))

	for _, v := range g.Edges {

		// Weight of zero means no connection so we skip.
		if v.Weight == 0.0 {
			continue
		}

		// Skip after adding self transitions.
		if v.From == v.To {
			ng.appendEdge(v.FromName, v.FromName, v.Weight)
			continue
		}

		// Insert edge to and from new context-dependent state.
		cdName := v.FromName + "-" + v.ToName
		cdNodes[cdName] = true
		ng.appendEdge(v.FromName, cdName, v.Weight)
		ng.appendEdge(cdName, v.ToName, 1.0)

		// Self transition for new state.
		ng.appendEdge(cdName, cdName, 1.0)
	}
	ng.createNodes()

	// Mark context dependent nodes using an attribute.
	for k, _ := range cdNodes {
		ng.Nodes[k].Atts = make(map[string]interface{})
		ng.Nodes[k].Atts["cd"] = true
	}

	if glog.V(3) {
		glog.Infof("Graph Name: %s", ng.Name)
		for k, v := range ng.Edges {
			glog.Infof("Edge %2d:%+v", k, *v)
		}
	}
	return
}

func (graph *Graph) String() string {

	var buffer bytes.Buffer

	fmt.Fprintf(&buffer, "Graph Name: %s\n", graph.Name)

	for k, v := range graph.Nodes {
		fmt.Fprintf(&buffer, "Node [%s], Atts: [%v]\n", k, v.Atts)
	}

	for k, v := range graph.Edges {
		fmt.Fprintf(&buffer, "Edge %2d: from [%s] to [%s] weight [%.2f].\n", k, v.FromName, v.ToName, v.Weight)
	}

	return buffer.String()
}

// Sort Nodes.

type Nodes []*Node

func (s Nodes) Len() int      { return len(s) }
func (s Nodes) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

// ByName implements sort.Interface by providing Less and using the Len and
type ByName struct{ Nodes }

func (s ByName) Less(i, j int) bool { return s.Nodes[i].Name < s.Nodes[j].Name }
