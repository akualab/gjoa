// Copyright (c) 2015 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

import "testing"

var (
	words = []string{"HELLO", "WORLD"}
	dict  = map[string][]string{"HELLO": []string{"HH", "AH0", "L", "OW1"},
		"WORLD": []string{"W", "ER1", "L", "D"}}
)

func TestAssigner(t *testing.T) {

	var a DirectAssigner
	names := a.Assign(words)

	for k, v := range words {
		if v != names[k] {
			t.Fatalf("direct assigner failed - word [%s] does not match name [%s]", v, names[k])
		}
	}

	var b MapAssigner = dict
	names = b.Assign(words)
	expected := []string{"HH", "AH0", "L", "OW1", "W", "ER1", "L", "D"}
	for i, name := range names {
		if name != expected[i] {
			t.Fatalf("map assigner failed - subword [%s] does not match expected [%s]", name, expected[i])
		}
	}
}
