// Copyright (c) 2015 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package model

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/akualab/gjoa"
)

func TestValid(t *testing.T) {
	var root *ANode
	a := &ANode{Start: 0, End: 3}
	b := &ANode{Start: 3, End: 8}
	c := &ANode{Start: 8, End: 10}
	root = &ANode{Start: 0, End: 10, Children: []*ANode{a, b, c}}
	if !root.IsValid() {
		t.Fatal("expected alignment tree to be valid but is not")
	}

	x := &ANode{Start: 3, End: 5}
	y := &ANode{Start: 5, End: 8}
	b.Children = []*ANode{x, y}
	if root.isBalanced() {
		t.Fatal("expected alignment tree to be unbalanced")
	}
	if !root.checkIntervals() {
		t.Fatal("expected alignment tree to have correct intervals but it doesn't")
	}
}

func makeTree() *ANode {
	root := NewANode(0, 15, "root", nil)
	a := root.AppendChild(3, "a")
	b := root.AppendChild(8, "b")
	c := root.AppendChild(15, "c")

	a.AppendChild(3, "a0")

	b.AppendChild(4, "b0")
	b.AppendChild(6, "b1")
	b.AppendChild(8, "b2")

	c.AppendChild(11, "c0")
	c.AppendChild(15, "c1")
	return root
}
func TestAppend(t *testing.T) {

	root := makeTree()
	if !root.IsValid() {
		t.Fatal("alignment tree is invalid")
	}

	level := root.Level()
	if level != 2 {
		t.Fatalf("expected level 2, got level %d", level)
	}

	for _, child := range root.Children {
		level := child.Level()
		if level != 1 {
			t.Fatalf("expected level 1, got level %d", level)
		}
	}

	al := root.Alignment()
	t.Log(al)
	tree := al.Tree()
	t.Log(tree.checkIntervals())
	t.Log(tree.isBalanced())
	t.Log(tree.IsValid())
	t.Log(tree)
}

func TestWriteRead(t *testing.T) {

	root := makeTree()
	al := root.Alignment()
	fn := filepath.Join(os.TempDir(), "align.json")
	gjoa.WriteJSONFile(fn, al)
	t.Logf("Wrote to temp file: %s\n", fn)

	var al2 Alignment
	gjoa.ReadJSONFile(fn, &al2)
	t.Log(al2)

	for level := range al {
		if len(al[level]) != len(al2[level]) {
			t.Fatalf("length for level [%d] does not match - expected %d, got %d", level, len(al[level]), len(al2[level]))
		}
		for i, a := range al[level] {
			b := al2[level][i]
			if a.Start != b.Start || a.End != b.End || a.Name != b.Name {
				t.Fatalf("nodes don't match - expected %s, got %s", a, b)
			}
		}
	}
}
