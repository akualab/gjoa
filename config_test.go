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

	if config.HMM.TPGraphFilename != "topology.yaml" {
		t.Fatalf("TPGraphFilename is [%s]. Expected \"topology.yaml\".", config.HMM.TPGraphFilename)
	}

	if config.Vectors[0][1] != "b" {
		t.Fatalf(" config.Vectors[0][1] is [%s]. Expected \"b\".", config.Vectors[0][1])
	}

}

const config string = `
model: hmm
data_set: random.yaml
hmm:
  tp_graph: topology.yaml

vectors:
  - [a,b,c]
  - [the_truth]
  - [outlook day]
`
