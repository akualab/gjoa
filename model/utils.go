// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package model

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"

	"github.com/golang/glog"
)

// RandNormalVector returns a random observation.
func RandNormalVector(r *rand.Rand, mean, std []float64) []float64 {

	if len(mean) != len(std) {
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

// ObsElem is an observation vector.
type ObsElem struct {
	Value [][]float64 `json:"value"`
	Label string      `json:"label"`
	ID    string      `json:"id"`
}

// StreamObserver implements an observer to stream FloatObs objects.
// Not safe to use with multiple goroutines.
type StreamObserver struct {
	reader io.Reader
}

// NewStreamObserver creates a new StreamObserver.
// Values are read from a reader as json format.
func NewStreamObserver(reader io.Reader) (*StreamObserver, error) {
	so := &StreamObserver{
		reader: reader,
	}
	return so, nil
}

// ObsChan implements the ObsChan method for the observer interface.
func (so StreamObserver) ObsChan() (<-chan Obs, error) {

	obsChan := make(chan Obs, 1000)
	go func() {

		dec := json.NewDecoder(so.reader)
		for {
			var v ObsElem
			if err := dec.Decode(&v); err != nil {
				glog.Warning(err)
				return
			}
			// Check if this is a sequence.
			seqLen := len(v.Value)

			if seqLen == 1 {
				obsChan <- NewFloatObs(v.Value[0], SimpleLabel(v.Label))
			} else {
				obsChan <- NewFloatObsSequence(v.Value, SimpleLabel(v.Label), v.ID)
			}
		}
		close(obsChan)
	}()

	return obsChan, nil
}
