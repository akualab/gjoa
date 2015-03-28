// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package model

import (
	"math"
	"math/rand"
	"testing"

	"github.com/akualab/gjoa"
)

// Tests

func TestRandomIntFromDist(t *testing.T) {

	dist := []float64{0.1, 0.2, 0.3, 0.4}

	r := rand.New(rand.NewSource(33))
	res1 := make([]int, 100, 100)
	for range res1 {
		res1 = append(res1, RandIntFromDist(dist, r))
	}

	// Checking that experiments are repeatable.
	r = rand.New(rand.NewSource(33))
	res2 := make([]int, 100, 100)
	for range res2 {
		res2 = append(res2, RandIntFromDist(dist, r))
	}

	gjoa.CompareSliceInt(t, res1, res1, "sequence mismatch")

	r = rand.New(rand.NewSource(33))
	res := make(map[int]float64)
	var n float64 = 100000
	for i := 0.0; i < n; i++ {
		res[RandIntFromDist(dist, r)]++
	}

	actual := make([]float64, len(dist), len(dist))
	for k, v := range res {
		p := v / n
		t.Log(k, v, p)
		actual[k] = p
	}

	gjoa.CompareSliceFloat(t, dist, actual, "probs don't match, error in RandIntFromDist", 0.02)

	// same with log probs
	logDist := make([]float64, len(dist), len(dist))
	for k, v := range dist {
		logDist[k] = math.Log(v)
	}

	r = rand.New(rand.NewSource(33))
	res = make(map[int]float64)
	for i := 0.0; i < n; i++ {
		res[RandIntFromLogDist(logDist, r)]++
	}

	for k, v := range res {
		p := v / n
		t.Log(k, v, p)
		actual[k] = p
	}

	gjoa.CompareSliceFloat(t, dist, actual, "probs don't match, error in RandIntFromLogDist", 0.02)

}
