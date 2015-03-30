// Copyright (c) 2015 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package model

// ANode is an alignment node. Assumptions:
//  * Root node (no parent) vovers the full interval.
//  * A child node interval is included in the parent interval.
//  * Concatenation of all the children intervals matches exactly the parent interval.
type ANode struct {
	// Start time (inclusive)
	Start int
	// End time (exclusive)
	End int
	// Name of unit being aligned.
	Name string
	// Pointer alignment one level up.
	Parent *ANode
	// Pointers to child alignments one level down.
	Children []*ANode
}

// Valid returns true if the alignment tree is valid.
func (a *ANode) Valid() bool {

	if a.Children == nil {
		return true
	}
	start := a.Start
	end := a.End
	last := start
	for _, v := range a.Children {
		if !v.Valid() {
			return false
		} else {
			if v.Start != last {
				return false
			}
			last = v.End
		}
	}
	if last != end {
		return false
	}
	return true
}

// The Aligner interface provides access to time alignment information.
type Aligner interface {

	// Alignment info.
	Alignments() []ANode
}
