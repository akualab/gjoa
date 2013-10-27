package gjoa

import ()

type Config struct {
	DataSet  string   `yaml:"data_set" json:"data_set"`
	Model    string   `yaml:"model" json:"model"`
	HMM      HMM      `yaml:"hmm" json:"hmm"`
	GMM      GMM      `yaml:"gmm" json:"gmm"`
	Gaussian Gaussian `yaml:"gaussian" json:"gaussian"`
}

type HMM struct {
	TPGraphFilename string `yaml:"tp_graph,omitempty" json:"tp_graph,omitempty"`
	UpdateTP        bool   `yaml:"update_tp,omitempty" json:"update_tp,omitempty"`
	UpdateIP        bool   `yaml:"update_ip,omitempty" json:"update_ip,omitempty"`
	UseAlignments   bool   `yaml:"use_alignments,omitempty" json:"use_alignments,omitempty"`
	OutputDist      string `yaml:"output_distribution,omitempty" json:"output_distribution,omitempty"`
	GeneratorSeed   int64  `yaml:"generator_seed,omitempty" json:"generator_seed,omitempty"`
	GeneratorMaxLen int    `yaml:"generator_max_length,omitempty" json:"generator_max_length,omitempty"`
}

type GMM struct{}

type Gaussian struct{}
