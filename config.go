package gjoa

import (
	"io/ioutil"

	"launchpad.net/goyaml"
)

type Config struct {
	DataSet  string              `yaml:"data_set" json:"data_set"`
	ModelOut string              `yaml:"model_out" json:"model_out"`
	Vectors  map[string][]string `yaml:"vectors" json:"vectors"`
	Model    string              `yaml:"model" json:"model"`
	HMM      HMM                 `yaml:"hmm" json:"hmm"`
	GMM      GMM                 `yaml:"gmm" json:"gmm"`
	Gaussian Gaussian            `yaml:"gaussian" json:"gaussian"`
}

type HMM struct {
	GraphIn         string `yaml:"graph_in,omitempty" json:"graph_in,omitempty"`
	GraphOut        string `yaml:"graph_out,omitempty" json:"graph_out,omitempty"`
	CDState         bool   `yaml:"cd_state,omitempty" json:"cd_state,omitempty"`
	UpdateTP        bool   `yaml:"update_tp,omitempty" json:"update_tp,omitempty"`
	UpdateIP        bool   `yaml:"update_ip,omitempty" json:"update_ip,omitempty"`
	UseAlignments   bool   `yaml:"use_alignments,omitempty" json:"use_alignments,omitempty"`
	OutputDist      string `yaml:"output_distribution,omitempty" json:"output_distribution,omitempty"`
	GeneratorSeed   int64  `yaml:"generator_seed,omitempty" json:"generator_seed,omitempty"`
	GeneratorMaxLen int    `yaml:"generator_max_length,omitempty" json:"generator_max_length,omitempty"`
	HMMFile         string `yaml:"hmm_file,omitempty" json:"hmm_file,omitempty"`
}

type GMM struct{}

type Gaussian struct{}

// Read the gjoa configuration file.
func ReadConfig(filename string) (config *Config, err error) {

	var data []byte
	data, err = ioutil.ReadFile(filename)
	if err != nil {
		return
	}
	config = &Config{}
	err = goyaml.Unmarshal(data, config)
	if err != nil {
		return
	}
	return
}
