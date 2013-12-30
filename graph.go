package gjoa

import (
	"fmt"

	"github.com/akualab/graph"
)

// Inserts a node between each pair of connected nodes. Assigns a weight of 1.0 to
// the self-transition and the transition from the new node to the succesor node.
// Assigns weight to the inserted node sel arc and (1-weight) to the other arc.
// Fails if weight is not a value between 0 and 1.
func InsertNodes(g *graph.Graph, weight float64) (ng *graph.Graph, e error) {

	if weight > 1 || weight < 0 {
		e = fmt.Errorf("invalid weight, must be 0 < weight < 1. Got [%f]", weight)
		return
	}

	ng, e = g.Clone()
	if e != nil {
		return
	}
	for _, from := range ng.GetAll() {

		// use a map to hold custom attributes.
		// set inserted:false to original nodes.
		value := map[string]interface{}{"inserted": false}
		ng.Set(from.Key(), value)

		// Get succesors.
		succesors := from.GetSuccesors()
		for to, weight := range succesors {

			// Skip if self-transition.
			if from == to {
				continue
			}
			if !from.Disconnect(to) {
				e = fmt.Errorf("Unable to disconnect nodes.")
				return
			}

			// Create new node.
			iName := from.Key() + "-" + to.Key() // new name using neighbors' names.
			iValue := map[string]interface{}{"inserted": true}
			iNode := ng.Set(iName, iValue) // insert node.

			// Make connection to and from inserted node and self-transition
			from.Connect(iNode, weight)
			iNode.Connect(to, 1-weight)
			if weight > 0.0 {
				iNode.Connect(iNode, weight)
			}
		}
	}
	return
}

// Inserts self arcs to each node and assigns weight. self arc exists, it updates
// the weight.
func InsertSelfArcs(g *graph.Graph, weight float64) (ng *graph.Graph, e error) {

	ng, e = g.Clone()
	if e != nil {
		return
	}

	for _, node := range ng.GetAll() {
		node.Connect(node, weight)
	}
	return
}
