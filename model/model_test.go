// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package model

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"

	"github.com/akualab/gjoa"
	"github.com/golang/glog"
)

func TestMain(m *testing.M) {

	// Configure glog. Example to set debug level 6 for file viterbi.go and 3 for everythign else:
	// export GLOG_LEVEL=3
	// go test -v  -run TestTrainHmmGau -vmodule=viterbi=6 > /tmp/zzz
	flag.Set("alsologtostderr", "true")
	flag.Set("log_dir", "/tmp/log")
	level := os.Getenv("GLOG_LEVEL")
	if len(level) == 0 {
		level = "0"
	}
	flag.Set("v", level)
	glog.Info("glog debug level is: ", level)

	os.Exit(m.Run())
}

func TestStreamObserver(t *testing.T) {

	dim := 4
	maxSeqLen := 10
	numObs := 12
	bufSize := 1000000

	r := rand.New(rand.NewSource(666))

	// Test non sequence data.
	reader := makeObsData(r, numObs, dim, 1)
	data := reader.(*obsReader).data // use to assert
	obs, err := NewStreamObserver(reader)
	if err != nil {
		t.Fatal(err)
	}
	c, e2 := obs.ObsChan()
	if e2 != nil {
		t.Fatal(e2)
	}
	i := 0
	for v := range c {
		if len(v.Value().([]float64)) != len(data[i].Value[0]) {
			t.Fatalf("length mismatch - got %d, expected %d", len(v.Value().([]float64)), len(data[i].Value[0]))
		}
		for j, exp := range data[i].Value[0] {
			got := v.Value().([]float64)[j]
			if exp != got {
				t.Logf("got: %5.3f, expected: %5.3f", exp, got)
			}
		}
		i++
	}

	// Test sequence data.
	reader = makeObsData(r, numObs, dim, maxSeqLen)
	data = reader.(*obsReader).data // use to assert
	bufReader := bufio.NewReaderSize(reader, bufSize)
	obs, err = NewStreamObserver(bufReader)
	if err != nil {
		t.Fatal(err)
	}
	c, e2 = obs.ObsChan()
	if e2 != nil {
		t.Fatal(e2)
	}

	i = 0
	for v := range c {

		if len(v.Value().([][]float64)) != len(data[i].Value) {
			t.Fatalf("length mismatch - i:%d, got:%d, expected:%d", i, len(v.Value().([][]float64)), len(data[i].Value))
		}
		for j, exp := range data[i].Value {
			got := v.Value().([][]float64)[j]
			gjoa.CompareSliceFloat(t, exp, got, "value mismatch", 0.001)
		}
		i++
	}
}

type obsReader struct {
	data []ObsElem
	idx  int
}

func (or *obsReader) Read(p []byte) (int, error) {

	if len(p) == 0 {
		return 0, nil
	}
	if or.idx >= len(or.data) {
		return 0, io.EOF
	}
	b, err := json.Marshal(or.data[or.idx])
	if err != nil {
		return 0, err
	}
	or.idx++
	if len(b) > len(p) {
		return 0, fmt.Errorf("buffer too short b:%d > p:%d", len(b), len(p))
	}
	n := copy(p, b)
	return n, nil
}

func makeObsData(r *rand.Rand, numObs, dim, maxSeqLen int) io.Reader {

	var isSeq bool
	if maxSeqLen < 1 {
		maxSeqLen = 1
	}
	if maxSeqLen > 1 {
		isSeq = true
	}
	data := make([]ObsElem, numObs, numObs)
	for k := range data {
		seqLen := r.Intn(maxSeqLen) + 1
		data[k] = ObsElem{
			ID:    "test-" + strconv.Itoa(k),
			Label: "len_" + strconv.Itoa(seqLen),
			Value: randFloats(r, seqLen, dim),
			IsSeq: isSeq,
		}
	}
	return &obsReader{data: data}
}

func randFloats(r *rand.Rand, n, dim int) [][]float64 {

	s := make([][]float64, n, n)
	for i := range s {
		s[i] = make([]float64, dim, dim)
		for j := range s[i] {
			s[i][j] = r.Float64()
		}
	}

	return s
}

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
