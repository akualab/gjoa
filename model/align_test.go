// Copyright (c) 2015 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package model

import "testing"

func TestValid(t *testing.T) {
	var root *ANode
	a := &ANode{Start: 0, End: 3, Parent: root}
	b := &ANode{Start: 3, End: 8, Parent: root}
	c := &ANode{Start: 8, End: 10, Parent: root}
	root = &ANode{Start: 0, End: 10, Children: []*ANode{a, b, c}}
	if !root.Valid() {
		t.Fatal("expected alignment tree to be valid but is not")
	}

	x := &ANode{Start: 3, End: 4, Parent: b}
	y := &ANode{Start: 5, End: 8, Parent: b}
	b.Children = []*ANode{x, y}
	if root.Valid() {
		t.Fatal("expected alignment tree to be invalid but is valid")
	}
}
