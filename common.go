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
	"path/filepath"

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

// Fatal logs a fatal errors and exits.
func Fatal(err error) {
	if err != nil {
		glog.Fatal(err)
	}
}

// ReadJSON unmarshals json data from an io.Reader.
// The param "o" must be a pointer to an object.
func ReadJSON(r io.Reader, o interface{}) error {
	dec := json.NewDecoder(r)
	err := dec.Decode(o)
	if err != nil && err != io.EOF {
		return err
	}
	return nil
}

// ReadJSONFile unmarshals json data from a file.
func ReadJSONFile(fn string, o interface{}) error {

	f, err := os.Open(fn)
	if err != nil {
		return err
	}
	defer f.Close()
	return ReadJSON(f, o)
}

// WriteJSON writes an object to an io.Writer.
func WriteJSON(w io.Writer, o interface{}) error {

	enc := json.NewEncoder(w)
	err := enc.Encode(o)
	if err != nil {
		return err
	}
	return nil
}

// WriteJSONFile writes to a file.
func WriteJSONFile(fn string, o interface{}) error {

	e := os.MkdirAll(filepath.Dir(fn), 0755)
	if e != nil {
		return e
	}
	f, err := os.Create(fn)
	if err != nil {
		return err
	}
	defer f.Close()

	ee := WriteJSON(f, o)
	if ee != nil {
		return ee
	}
	return nil
}
