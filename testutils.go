// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gjoa

import (
	"math"
	"testing"
)

// Comparef64 returns true if |f2/f1-1| < tol.
func Comparef64(f1, f2, tol float64) bool {
	avg := math.Abs(f1+f2) / 2.0
	sErr := math.Abs(f2-f1) / (avg + 1)
	if sErr < tol {
		return true
	}
	return false
}

// CompareSliceFloat compares slices elementwise using Comparef64.
func CompareSliceFloat(t *testing.T, expected []float64, actual []float64, message string, tol float64) {
	for i, _ := range expected {
		if !Comparef64(expected[i], actual[i], tol) {
			t.Errorf("[%s]. Expected: [%f], Got: [%f]",
				message, expected[i], actual[i])
		}
	}
}

// CompareFloats compares floats using Comparef64.
func CompareFloats(t *testing.T, expected float64, actual float64, message string, tol float64) {
	if !Comparef64(expected, actual, tol) {
		t.Errorf("[%s]. Expected: [%f], Got: [%f]",
			message, expected, actual)
	}
}

// CompareSliceInt compares two slices of ints elementwise.
func CompareSliceInt(t *testing.T, expected []int, actual []int, message string) {
	for i, _ := range expected {
		if expected[i] != actual[i] {
			t.Errorf("[%s]. Expected: [%d], Got: [%d]",
				message, expected[i], actual[i])
		}
	}
}

// CheckError calls Fatal if error is not nil.
func CheckError(t *testing.T, e error) {

	if e != nil {
		t.Fatal(e)
	}
}
