// Copyright (c) 2014 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gjoa

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestConfig(t *testing.T) {

	// Prepare dirs.
	tmpDir := os.TempDir()
	os.MkdirAll(tmpDir+"test-config", 0755)

	// Create file list yaml file.
	fn := tmpDir + "config.yaml"
	t.Logf("Config File: %s.", fn)
	err := ioutil.WriteFile(fn, []byte(config), 0644)
	CheckError(t, err)

	// Read config.
	config, e := ReadConfig(fn)
	CheckError(t, e)

	// Check Config content.
	t.Logf("Config: %+v", config)

	if config.Model != "hmm" {
		t.Fatalf("Model is [%s]. Expected \"HMM\".", config.Model)
	}

	if config.DataSet != "random.yaml" {
		t.Fatalf("DataSet is [%s]. Expected \"random.yaml\".", config.DataSet)
	}

	if config.HMM.GraphIn != "topology.yaml" {
		t.Fatalf("GraphIn is [%s]. Expected \"topology.yaml\".", config.HMM.GraphIn)
	}

	if config.Vectors["features"][1] != "b" {
		t.Fatalf(" config.Vectors[\"features\"][1] is [%s]. Expected \"b\".", config.Vectors["features"][1])
	}

}

const config string = `
model: hmm
data_set: random.yaml
vectors: {features: [a,b,c], class: [room]}
hmm:
  graph_in: topology.yaml
`
