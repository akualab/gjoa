package gjoa

import ()

type Config struct {
	Model string `yaml:"model" json:"model"`

	HMM HMM

	GMM GMM

	Gaussian Gaussian
}

type HMM struct {
	UpdateTP        bool   `yaml:"update_tp,omitempty" json:"update_tp,omitempty"`
	UpdateIP        bool   `yaml:"update_ip,omitempty" json:"update_ip,omitempty"`
	OutputDist      string `yaml:"output_distribution,omitempty" json:"output_distribution,omitempty"`
	GeneratorSeed   int64  `yaml:"generator_seed,omitempty" json:"generator_seed,omitempty"`
	GeneratorMaxLen int    `yaml:"generator_max_length,omitempty" json:"generator_max_length,omitempty"`
}

type GMM struct{}

type Gaussian struct{}
