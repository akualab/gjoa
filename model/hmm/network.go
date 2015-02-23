// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

import (
	"fmt"

	"github.com/akualab/gjoa/model"
	"github.com/gonum/graph"
	dg "github.com/gonum/graph/concrete"
)

// State is an HMM state. We try to hide the graph implementation details as
// much as possible. This will make it easier to build and use HMM networks and
// and make changes to the implementation in the future without breaking applications.
type State struct {
	α       []float64
	β       []float64
	isEntry bool
	isExit  bool
	node    graph.Node
	model   model.Scorer
	net     *Network
}

// ID returns the unique state id.
func (s *State) ID() int {
	return s.node.ID()
}

// Network is the core HMM object composed of states and arcs.
type Network struct {
	graph  *dg.DirectedGraph
	states []*State
	entry  *State
	exit   *State
}

// NewNetwork creates new HMM network.
func NewNetwork() *Network {

	return &Network{
		graph:  dg.NewDirectedGraph(),
		states: make([]*State, 0, cap),
	}
}

// NumStates returns the total number of states in the network.
func (net *Network) NumStates() int {
	return len(net.graph.NodeList())
}

// States returns the list of states in the network.
func (net *Network) States() []*State {
	return net.states
}

// AddEntryState adds a non-emitting state to the network.
// This must be the only entry state. Will panic otherwise.
func (net *Network) AddEntryState() (*State, error) {
	return net.addState(nil, true, false)
}

// AddExitState adds a non-emitting state to the network.
// This must be the only exit state. Will panic otherwise.
func (net *Network) AddExitState() (*State, error) {
	return net.addState(nil, false, true)
}

// AddState adds a state to the network. A state must have an
// associated output distribution of type *model.Scorer.
func (net *Network) AddState(dist model.Scorer) (*State, error) {
	return net.addState(dist, false, false)
}

func (net *Network) addState(dist model.Scorer, isEntry, isExit bool) (*State, error) {

	var alpha, beta []float64

	switch {
	case isEntry && isExit:
		return nil, fmt.Errorf("state cannot be simultaneously be entry and exit")
	case isEntry && net.entry != nil:
		return nil, fmt.Errorf("attempted to insert more than one entry state to network")
	case isExit && net.exit != nil:
		return nil, fmt.Errorf("attempted to insert more than one exit state to network")
	case !isEntry && !isExit:
		alpha = make([]float64, 0, cap)
		beta = make([]float64, 0, cap)
	}

	state := &State{
		α:       alpha,
		β:       beta,
		isEntry: isEntry,
		isExit:  isExit,
		node:    net.graph.NewNode(),
		net:     net,
		model:   dist,
	}

	if isEntry {
		net.entry = state
	}
	if isExit {
		net.exit = state
	}
	net.states = append(net.states, state)
	return state, nil
}

// AddArc adds a weighted arc connecting two nodes.
func (net *Network) AddArc(from, to *State, weight float64) {
	net.graph.AddDirectedEdge(dg.Edge{H: from.node, T: to.node}, weight)
}

// Validate checks that network topology is valid.
func (net *Network) Validate() error {

	if net.entry == nil {
		return fmt.Errorf("missing entry state")
	}
	if net.exit == nil {
		return fmt.Errorf("missing entry state")
	}

	// TODO: check network is left to right using topological order.

	return nil
}

// A Chain is a composite network where an exit node of a subnet is
// connected to the entry node of the following subnet.
type Chain []*Network

// NewChain creates a new composite network of type Chain.
func NewChain(nets ...*Network) Chain {

	return nets
}
