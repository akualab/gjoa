// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package model

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/gonum/floats"
)

// RandNormalVector returns a random observation.
func RandNormalVector(mean, std []float64, r *rand.Rand) []float64 {

	if !floats.EqualLengths(mean, std) {
		panic(fmt.Errorf("Cannot generate random vectors because length of mean [%d] and std [%d] don't match.",
			len(mean), len(std)))
	}
	vector := make([]float64, len(mean))
	for i, _ := range mean {
		v := r.NormFloat64()*std[i] + mean[i]
		vector[i] = v
	}

	return vector
}

// RandIntFromDist randomly selects an item using a discrete PDF.
// TODO: This is not optimal but should work for testing.
func RandIntFromDist(dist []float64, r *rand.Rand) int {
	N := len(dist)
	if N == 0 {
		panic("Error prob distribution has len 0")
	}
	ran := r.Float64()
	cum := 0.0
	for i := 0; i < N; i++ {
		cum = cum + dist[i]
		if ran < cum {
			return i
		}
	}
	return N - 1
}

// RandIntFromLogDist random selects an item using a discrete PDF.
// Slice dist contains log probabilities.
func RandIntFromLogDist(dist []float64, r *rand.Rand) int {
	N := len(dist)
	if N == 0 {
		panic("Error prob distribution has len 0")
	}
	ran := r.Float64()
	cum := 0.0
	for i := 0; i < N; i++ {
		cum = cum + math.Exp(dist[i])
		if ran < cum {
			return i
		}
	}
	return N - 1
}
