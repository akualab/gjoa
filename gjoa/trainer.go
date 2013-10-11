package main

import (
	"flag"
	"fmt"
	"github.com/akualab/gjoa"
	"github.com/golang/glog"
	"io/ioutil"
	"launchpad.net/goyaml"
	"os"
	"path"
)

var cmdTrainer = &Command{
	Run:       trainer,
	UsageLine: "train [options]",
	Short:     "runs trainer",
	Long: `
runs trainer.

You must provide a config file. The default name is "config.yaml".
A sample config file will look like this:

model: hmm
hmm:
  - update_tp: false
  - update_ip: false
  - output_distribution: gaussian

ex:
 $ gjoa train
`,
	Flag: *flag.NewFlagSet("gjoa-trainer", flag.ExitOnError),
}

func init() {
	addTrainerFlags(cmdTrainer)
}

var eid string // experiment id
var dir string
var model string
var updateTP bool
var updateIP bool
var outputDistribution string
var configFilename string

func addTrainerFlags(cmd *Command) {

	defaultDir, err := os.Getwd()
	if err != nil {
		defaultDir = os.TempDir()
	}
	defaultEID := path.Base(defaultDir)

	// Common.
	cmd.Flag.StringVar(&dir, "dir", defaultDir, "the project dir")
	cmd.Flag.StringVar(&eid, "eid", defaultEID, "the experiment id")
	cmd.Flag.StringVar(&configFilename, "config-file", "gjoa.yaml", "the trainer config file")

	// Selects a model.
	cmd.Flag.StringVar(&model, "model", "gaussian", "select a model to train {gaussian, gmm, hmm}")

	// Single Gaussian flags.

	// Gaussian mixture flags.

	// HMM flags.
	cmd.Flag.BoolVar(&updateTP, "hmm-update-tp", true, "update HMM transition probabilities, overwrites config file when set")
	cmd.Flag.BoolVar(&updateIP, "hmm-update-ip", true, "update HMM initial probabilities, overwrites config file when set")
	cmd.Flag.StringVar(&outputDistribution, "hmm-output-distribution", "gaussian", "HMM state output distribution {gaussian, gmm}")

}

func trainer(cmd *Command, args []string) {

	// Read config file.
	fn := fmt.Sprintf("%s%c%s", dir, os.PathSeparator, configFilename)
	data, err := ioutil.ReadFile(fn)
	gjoa.Fatal(err)
	config := gjoa.Config{}
	err = goyaml.Unmarshal(data, &config)
	gjoa.Fatal(err)

	// Overwrite config when a flag is set.
	// TODO: Is this the best way to do this?
	// I included this to show how it can be done. We probably want to have most
	// params in config and a few in command line to run experiments.
	cmd.Flag.Visit(func(f *flag.Flag) {

		switch f.Name {

		case "hmm-update-tp":
			config.HMM.UpdateTP = updateTP
		case "hmm-update-ip":
			config.HMM.UpdateIP = updateIP
		default:
			goto DONE
		}
		glog.Infof("Overwriting config using flag [%s] with value [%v]", f.Name, f.Value)
	DONE:
	})

	// Print config.
	glog.Infof("Read configuration:\n%+v", config)

	// Select model.
	glog.Infof("Training Model: %s.", config.Model)

	// Select the models, here do validation, bookkeeping, etc.
	switch model {
	case "gaussian":

	case "gmm":

	case "hmm":

		glog.Infof("Output distribution: %s.", config.HMM.OutputDist)
		switch outputDistribution {

		case "gaussian":

		case "gmm":

		default:
			glog.Fatalf("Unknown output distribution: %s.", outputDistribution)
		}

	default:
		glog.Fatalf("Unknown model: %s.", model)
	}
}
