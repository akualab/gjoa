package gjoa

import (
	"io/ioutil"
	"github.com/golang/glog"

	"launchpad.net/goyaml"
)

type Config struct {
	DataSet  string              `yaml:"data_set" json:"data_set"`
	ModelOut string              `yaml:"model_out" json:"model_out"`
	ModelIn  string              `yaml:"model_in" json:"model_in"`
	Vectors  map[string][]string `yaml:"vectors" json:"vectors"`
	Model    string              `yaml:"model" json:"model"`
	HMM      HMM                 `yaml:"hmm" json:"hmm"`
	GMM      GMM                 `yaml:"gmm" json:"gmm"`
	Gaussian Gaussian            `yaml:"gaussian" json:"gaussian"`

	ResultsFile   string `yaml:"results_file,omitempty" json:"results_file,omitempty"`
	ScoreFile     string `yaml:"score_file,omitempty" json:"score_file,omitempty"`
	NumIterations int    `yaml:"num_terations,omitempty" json:"num_iterations,omitempty"`
}

type HMM struct {
	GraphIn            string  `yaml:"graph_in,omitempty" json:"graph_in,omitempty"`
	GraphOut           string  `yaml:"graph_out,omitempty" json:"graph_out,omitempty"`
	AlignIn            string  `yaml:"align_in,omitempty" json:"align_in,omitempty"`
	AlignOut           string  `yaml:"align_out,omitempty" json:"align_out,omitempty"`
	CDState            bool    `yaml:"cd_state,omitempty" json:"cd_state,omitempty"`
	NormalizeWeights   bool    `yaml:"normalize_weights,omitempty" json:"normalize_weights,omitempty"`
	LogWeights         bool    `yaml:"log_weights,omitempty" json:"log_weights,omitempty"`
	SelfTransition     float64 `yaml:"self_transition,omitempty" json:"self_transition,omitempty"`
	UpdateTP           bool    `yaml:"update_tp,omitempty" json:"update_tp,omitempty"`
	UpdateIP           bool    `yaml:"update_ip,omitempty" json:"update_ip,omitempty"`
	UseAlignments      bool    `yaml:"use_alignments,omitempty" json:"use_alignments,omitempty"`
	ExpandedGraph      bool    `yaml:"expanded_graph,omitempty" json:"expanded_graph,omitempty"`
	OutputDist         string  `yaml:"output_distribution,omitempty" json:"output_distribution,omitempty"`
	GeneratorSeed      int64   `yaml:"generator_seed,omitempty" json:"generator_seed,omitempty"`
	GeneratorMaxLen    int     `yaml:"generator_max_length,omitempty" json:"generator_max_length,omitempty"`
	HMMFile            string  `yaml:"hmm_file,omitempty" json:"hmm_file,omitempty"`
	SplitDash          bool    `yaml:"split_dash,omitempty" json:"split_dash,omitempty"`
	TwoStateCollection bool    `yaml:"two_state_collection,omitempty" json:"two_state_collection,omitempty"`
	JoinCollection     bool    `yaml:"join_collection,omitempty" json:"join_collection,omitempty"`
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
	glog.Infof("config:\n%s\n\n", config)
	return
}

func (c *Config) String() string {

	d, err := goyaml.Marshal(c)
	if err != nil {
		glog.Fatal(err)
	}
	return string(d)
}
