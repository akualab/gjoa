package gjoa

import (
	"github.com/golang/glog"
	"io"
	"io/ioutil"
	"launchpad.net/goyaml"
	"os"
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
