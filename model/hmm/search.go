// Copyright (c) 2015 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

import (
	"math"
	"strconv"

	"github.com/akualab/gjoa/model"
	"github.com/akualab/graph"
	"github.com/golang/glog"
)

// Value is the value type associated with graph nodes. Must implement the
// graph.Viterbier interface.
type nodeValue struct {
	scorer      model.Modeler
	entry, exit bool
}

// Score implements the graph.Viterbier interface.
// The argument x must be of type []float64.
func (nv nodeValue) Score(x interface{}) float64 {
	if nv.scorer == nil {
		return 0 // non-emitting node.
	}
	o := model.NewFloatObs(x.([]float64), model.SimpleLabel(""))
	return nv.scorer.LogProb(o)
}

func (nv nodeValue) IsNull() bool {
	if nv.scorer == nil {
		return true
	}
	return false
}

// SearchGraph creates a graph based on a set of HMM networks.
func (set *Set) SearchGraph() *graph.Graph {

	glog.Infof("building search graph using %d hmms", set.size())
	g := graph.New()
	for _, m := range set.Nets {
		glog.Infof("adding model [%s]", m.Name)
		err := g.Add(m.Graph())
		if err != nil {
			glog.Fatalf("failed to build search graph with error: ", err)
		}
	}

	// Log prob of model in search graph.
	logp := -math.Log(float64(set.size()))

	// Start node. Only node with no predescessors.
	start := g.Set("start", nodeValue{})

	// End node. Only no with no successors.
	end := g.Set("end", nodeValue{})

	// Backoff node connects all networks exit states to all entry states.
	backoff := g.Set("backoff", nodeValue{})

	nodes := g.GetAll()

	for _, n := range nodes {
		val := n.Value().(nodeValue)

		if val.entry {
			// backoff to entry
			backoff.Connect(n, logp)
		}

		if val.exit {
			// exit to backoff
			n.Connect(backoff, 0)
		}
	}

	// backoff to end
	backoff.Connect(end, 0)
	// start to backoff
	start.Connect(backoff, 0)

	// Normalize all transition weights.
	// TODO

	glog.Infof("search graph created - num nodes: %d", g.Len())
	glog.V(3).Info("search graph:\n", g)
	return g
}

// Graph converts an HMM Net object to a graph where states are nodes. Like
// an HMM net, the graph contains a single entry and exit non-emmiting state.
func (m *Net) Graph() *graph.Graph {

	g := graph.New()

	// Create nodes.
	for i := 0; i < m.ns; i++ {
		var entry, exit bool
		if i == 0 {
			entry = true
		}
		if i == m.ns-1 {
			exit = true
		}
		key := m.Name + "-" + strconv.FormatInt(int64(i), 10)
		nv := nodeValue{scorer: m.B[i], entry: entry, exit: exit}
		g.Set(key, nv)
	}

	// Connect nodes.
	for i := 0; i < m.ns; i++ {
		fromKey := m.Name + "-" + strconv.FormatInt(int64(i), 10)
		for j := i; j < m.ns; j++ {
			if m.A.At(i, j) > math.Inf(-1) {
				toKey := m.Name + "-" + strconv.FormatInt(int64(j), 10)
				g.Connect(fromKey, toKey, m.A.At(i, j))
			}
		}
	}
	return g
}
