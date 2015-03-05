// Copyright (c) 2015 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmm

// Assigner assigns a sequence of hmm model names to a sequence of labels.
// This interface hides the details of how the assignment is done. For example,
// a trivial assigner may assign the label name to the model name:
//
//   Input labels:       []string{"a", "b", "c"}
//   Output model names: []string{"a", "b", "c"}
//
// or it may use a dictionary as is the case in speech recognition:
//
//   Input labels:       []string{"HELLO","WORLD"}
//   Output model names: []string{"HH","AH0","L","OW1","W","ER1","L","D"}
//
type Assigner interface {
	Assign(labels []string) (modelNames []string)
}

// DirectAssigner implements the Assigner interface.
// Model names correspond one-to-one to the label names.
type DirectAssigner struct{}

// Assign returns a sequence of model names.
func (a DirectAssigner) Assign(labels []string) []string {

	return append([]string(nil), labels...)
}

// MapAssigner implements the Assigner interface.
// Labes are mapped using a dictionary.
type MapAssigner map[string][]string

// Assign returns a sequence of model names.
func (a MapAssigner) Assign(labels []string) []string {

	var names []string
	for _, word := range labels {
		names = append(names, a[word]...)
	}
	return names
}
