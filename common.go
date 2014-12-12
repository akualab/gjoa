// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gjoa

import (
	"bufio"
	"encoding/json"
	"io"
	"os"

	"github.com/golang/glog"
)

type Result struct {
	BatchID string   `json:"batchid"`
	Ref     []string `json:"ref,omitempty"`
	Hyp     []string `json:"hyp,omitempty"`
}

// Write a collection of results to a file.
func WriteResults(results map[string]*Result, fn string) error {

	f, e := os.Create(fn)
	if e != nil {
		return e
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, v := range results {
		e := enc.Encode(v)
		if e != nil {
			return e
		}
	}
	return nil
}

// Read a results file.
func ReadResults(fn string) (results map[string]*Result, e error) {

	var f *os.File
	f, e = os.Open(fn)
	if e != nil {
		return
	}
	defer f.Close()
	reader := bufio.NewReader(f)

	results = make(map[string]*Result)

	for {
		var b []byte
		b, e = reader.ReadBytes('\n')
		if e == io.EOF {
			e = nil
			return
		}
		if e != nil {
			return
		}

		result := new(Result)
		e = json.Unmarshal(b, result)
		if e != nil {
			return
		}
		results[result.BatchID] = result
	}
	return
}

func Fatal(err error) {
	if err != nil {
		glog.Fatal(err)
	}
}
