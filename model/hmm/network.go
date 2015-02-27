// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

import (
	"fmt"
	"math"

	"github.com/akualab/gjoa/model"
	"github.com/akualab/narray"
	"github.com/golang/glog"
	"github.com/gonum/graph"
	dg "github.com/gonum/graph/concrete"
)

// State is an HMM state. We try to hide the graph implementation details as
// much as possible. This will make it easier to build and use HMM networks and
// and make changes to the implementation in the future without breaking applications.
type State struct {
	isEntry bool
	isExit  bool
	node    graph.Node
	model   model.Scorer
	net     *Network
}

// ID returns the unique state id.
func (s *State) ID() int {
	return (s.node.ID() - 1)
}

// Model returns the model associated with this state.
func (s *State) Model() model.Scorer {
	return s.model
}

// IsEmitting returns true if this is an emitting state. (Has a model.)
func (s *State) IsEmitting() bool {
	if s.isEntry || s.isExit {
		return false
	}
	return true
}

// Succesors returns the succesors of this state.
func (s *State) Succesors() []*State {
	//succ := s.net.graph.Successors(s.node)
	return nil
}

// LogProb returns the model associated with this state.
func (s *State) LogProb(obs model.Obs) float64 {
	if s.IsEmitting() {
		return s.model.LogProb(obs)
	}
	return math.Inf(-1)
}

// Network is the core HMM object composed of states and arcs.
type Network struct {
	name   string
	graph  *dg.DirectedGraph
	states []*State
	entry  *State
	exit   *State
	α      *narray.NArray
	β      *narray.NArray
}

// NewNetwork creates new HMM network.
func NewNetwork(name string) *Network {

	return &Network{
		graph:  dg.NewDirectedGraph(),
		states: make([]*State, 0, cap),
		name:   name,
	}
}

// Name of this network.
func (net *Network) Name() string {
	return net.name
}

// NumStates returns the total number of states in the network.
func (net *Network) NumStates() int {
	return len(net.graph.NodeList())
}

// States returns the list of states in the network.
func (net *Network) States() []*State {
	return net.states
}

// EmitStates returns a list of all emitting states.
func (net *Network) EmitStates() []*State {
	list := []*State{}
	for _, s := range net.states {
		if !s.isEntry && !s.isExit {
			list = append(list, s)
		}
	}
	return list
}

// State returns the state given the id.
func (net *Network) State(id int) *State {
	return net.states[id]
}

// EntryState returns the entry state.
func (net *Network) EntryState() *State {
	return net.entry
}

// ExitState returns the exit state.
func (net *Network) ExitState() *State {
	return net.exit
}

// AddEntryState adds a non-emitting state to the network.
// This must be the only entry state. Will panic otherwise.
func (net *Network) AddEntryState() *State {
	s, e := net.addState(nil, true, false)
	if e != nil {
		panic(e)
	}
	return s
}

// AddExitState adds a non-emitting state to the network.
// This must be the only exit state. Will panic otherwise.
func (net *Network) AddExitState() *State {
	s, e := net.addState(nil, false, true)
	if e != nil {
		panic(e)
	}
	return s
}

// AddState adds a state to the network. A state must have an
// associated output distribution of type *model.Scorer.
func (net *Network) AddState(dist model.Scorer) *State {
	s, e := net.addState(dist, false, false)
	if e != nil {
		panic(e)
	}
	return s

}

func (net *Network) addState(dist model.Scorer, isEntry, isExit bool) (*State, error) {

	switch {
	case isEntry && isExit:
		return nil, fmt.Errorf("state cannot be simultaneously be entry and exit")
	case isEntry && net.entry != nil:
		return nil, fmt.Errorf("attempted to insert more than one entry state to network")
	case isExit && net.exit != nil:
		return nil, fmt.Errorf("attempted to insert more than one exit state to network")
	}

	state := &State{
		isEntry: isEntry,
		isExit:  isExit,
		node:    net.graph.NewNode(),
		net:     net,
		model:   dist,
	}

	stateType := "emitting"
	if isEntry {
		stateType = "entry"
		net.entry = state
	}
	if isExit {
		stateType = "exit"
		net.exit = state
	}
	net.states = append(net.states, state)
	glog.Infof("added %s state with id [%d] to network [%s]", stateType, state.node.ID(), net.name)
	return state, nil
}

// AddArc adds a weighted arc connecting two nodes.
func (net *Network) AddArc(from, to *State, weight float64) {
	net.graph.AddDirectedEdge(dg.Edge{H: from.node, T: to.node}, weight)
}

// ArcWeight returns the arc weight.
// Returns -Inf if the arc does not exist.
func (net *Network) ArcWeight(from, to *State) float64 {
	edge := net.graph.EdgeTo(from.node, to.node)
	if edge == nil {
		return math.Inf(-1)
	}
	return net.graph.Cost(edge)
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

// Init accumulators for nobs observations.
func (net *Network) Init(nobs int) {

	glog.Infof("initializing network [%s] with [%d] states and [%d] observations", net.name, net.NumStates(), nobs)
	nstates := len(net.graph.NodeList())
	net.α = narray.New(nstates, nobs)
	net.α.SetValue(math.Inf(-1))
	net.β = narray.New(nstates, nobs)
	net.β.SetValue(math.Inf(-1))
}

// A Chain is a composite network where an exit node of a subnet is
// connected to the entry node of the following subnet.
type Chain []*Network

// NewChain creates a new composite network of type Chain.
func NewChain(nets ...*Network) Chain {

	return nets
}
