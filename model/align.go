// Copyright (c) 2015 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package model

import (
	"bytes"
	"fmt"

	"github.com/akualab/gjoa"
)

// The Aligner interface provides access to alignment information at various levels.
type Aligner interface {

	// Alignment info for all available levels.
	Alignment() []*ANode
}

// Alignment represents alignments as a sequence of ANodes by level.
type Alignment [][]*ANode

// ANode is an alignment node. Assumptions:
//  * Root node (no parent) covers the full interval.
//  * A child node interval is included in the parent interval.
//  * Concatenation of children intervals must match exactly the parent interval.
//  * Tree must be balanced.
type ANode struct {
	// Start index (inclusive)
	Start int `json:"s"`
	// End index (exclusive)
	End int `json:"e"`
	// Name of unit being aligned.
	Name string `json:"n"`
	// Value is an arbitrary object associated to an interval.
	Value interface{} `json:"v"`
	// Pointers to child alignments one level down.
	Children []*ANode `json:"-"`
}

// NewANode creates a new ANode.
func NewANode(start, end int, name string, value interface{}) *ANode {
	return &ANode{Start: start, End: end, Name: name, Value: value, Children: []*ANode{}}
}

// Copy copies an ANode.
// Children and Value fields are not copied.
func (a *ANode) Copy() *ANode {

	return &ANode{
		Start:    a.Start,
		End:      a.End,
		Name:     a.Name,
		Children: []*ANode{},
	}
}

// Alignment returns alignments by level. The first slice index corresponds to a level
// from 0 to max depth. The second index is the nth contiguous ANode at that level.
// The ANodes are NOT copied. Make explicit copies if you need an independent set of ANodes.
func (a *ANode) Alignment() Alignment {

	bl := &byLevel{data: Alignment{}}
	root := a.Copy() // dummy root
	root.Children = []*ANode{a}
	root.nav(bl)
	return bl.data
}

type byLevel struct {
	data Alignment
}

func (a *ANode) nav(ss *byLevel) (*ANode, int) {
	level := 0
	if len(a.Children) == 0 {
		return a, 0
	}
	for _, v := range a.Children {
		var child *ANode
		child, level = v.nav(ss)
		if level+1 > len(ss.data) {
			ss.data = append(ss.data, []*ANode{})
		}
		ss.data[level] = append(ss.data[level], child)
	}
	return a, level + 1
}

// Level retrurns the level of the ANode. Level zero corresponds to the leaves.
func (a *ANode) Level() int {
	if len(a.Children) == 0 {
		return 0
	}
	return a.Children[0].Level() + 1
}

// AppendChild creates a new ANode and appends to the receiver node.
// The start index of the new node equals the end index of the last child.
// Will panic if end < start OR end > parent's end.
func (a *ANode) AppendChild(end int, name string) *ANode {

	start := a.Start
	if len(a.Children) != 0 {
		start = a.Children[len(a.Children)-1].End
	}
	if end < start {
		err := fmt.Errorf("end index [%d] must be greater than start index [%d]", end, start)
		panic(err)
	}
	if end > a.End {
		err := fmt.Errorf("end index [%d] must be less or equal than parent end index [%d]", end, a.End)
		panic(err)
	}

	child := &ANode{
		Start: start,
		End:   end,
		Name:  name,
	}

	a.Children = append(a.Children, child)
	return child
}

// IsValid returns true if the alignment subtree is valid.
func (a *ANode) IsValid() bool {
	return a.checkIntervals() && a.isBalanced()
}

func (a *ANode) checkIntervals() bool {

	if a.Children == nil {
		return true
	}
	start := a.Start
	end := a.End
	last := start
	for _, v := range a.Children {
		if !v.checkIntervals() {
			return false
		}
		if v.Start != last {
			return false
		}
		last = v.End
	}
	if last != end {
		return false
	}
	return true
}

func (a *ANode) isBalanced() bool {

	var prev int
	for i, child := range a.Children {
		level := child.Level()
		if i == 0 {
			prev = level
			continue
		}
		if prev != level {
			return false
		}
		prev = level
	}
	return true
}

// ToJSON returns a json string.
func (a *ANode) ToJSON() (string, error) {
	var b bytes.Buffer
	err := gjoa.WriteJSON(&b, a)
	return b.String(), err
}

// String prints an ANode.
func (a *ANode) String() string {
	s, err := a.ToJSON()
	if err != nil {
		panic(err)
	}
	return s
}

// Tree converts an Alignment object to an alignment tree.
// ANodes are NOT copied from the alignment data, instead the Children
// fields are set to point to the children nodes.
func (a Alignment) Tree() *ANode {
	depth := len(a)
	for level := depth - 2; level >= 0; level-- {
		i := 0
		var parent *ANode
		for _, node := range a[level] {
			if i == 0 || (node.Start == parent.End) {
				parent = a[level+1][i]
				parent.Children = []*ANode{}
				i++
			}
			parent.Children = append(parent.Children, node)
		}
	}
	return a[depth-1][0]
}

// Level returns the list of intervals for a specific level.
func (a Alignment) Level(level int) []*ANode {
	return a[level]
}

// AlignLabels converts a slice of strings to an Alignment object.
// Consecutive elements with the same label are merged into an ANode.
func AlignLabels(labels []string) []*ANode {
	lastLabel := labels[0]
	anode := &ANode{Start: 0, Name: labels[0]}
	anodes := []*ANode{}
	for idx, v := range labels {
		if v != lastLabel {
			anode.End = idx
			anodes = append(anodes, anode)
			anode = &ANode{Start: idx, Name: v}
			lastLabel = v
		}
	}
	anode.End = len(labels)
	anodes = append(anodes, anode)

	return anodes
}

// ToJSON returns a json string.
func (a Alignment) ToJSON() (string, error) {
	var b bytes.Buffer
	err := gjoa.WriteJSON(&b, a)
	return b.String(), err
}

// String prints alignments.
func (a Alignment) String() string {
	s, err := a.ToJSON()
	if err != nil {
		panic(err)
	}
	return s
}
