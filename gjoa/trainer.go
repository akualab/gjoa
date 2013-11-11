package main

import (
	"flag"
	"fmt"
	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model"
	"github.com/akualab/gjoa/model/gaussian"
	"github.com/akualab/gjoa/model/hmm"
	"github.com/golang/glog"
	"io"
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

const (
	DEFAULT_CONFIG_FILE = "gjoa.yaml"
	DEFAULT_DATA_SET    = "train.yaml"
	DEFAULT_OUT_MODEL   = "gaussian"
	DEFAULT_MODEL       = "gaussian"
	DEFAULT_TP_FILE     = "tp.yaml"
)

var tpFilename string
var modelType string
var updateTP bool
var updateIP bool
var useAlignments bool
var outputDistribution string
var configFilename string
var dataSet string
var modelOutFilename string
var modelInFilename string

func addTrainerFlags(cmd *Command) {

	defaultDir, err := os.Getwd()
	if err != nil {
		defaultDir = os.TempDir()
	}
	defaultEID := path.Base(defaultDir)

	// Common.
	cmd.Flag.StringVar(&dir, "dir", defaultDir, "the project dir")
	cmd.Flag.StringVar(&eid, "eid", defaultEID, "the experiment id")
	cmd.Flag.StringVar(&configFilename, "config-file", DEFAULT_CONFIG_FILE, "the trainer config file")
	cmd.Flag.StringVar(&dataSet, "data-set", DEFAULT_DATA_SET, "the file with the list of data files")
	cmd.Flag.StringVar(&modelOutFilename, "model-out", "model-out.json", "output model filename")
	cmd.Flag.StringVar(&modelInFilename, "model-in", "model-in.json", "input model filename")

	// Selects a model.
	cmd.Flag.StringVar(&modelType, "model", DEFAULT_MODEL, "select a model to train {gaussian, gmm, hmm}")

	// Single Gaussian flags.

	// Gaussian mixture flags.

	// HMM flags.
	cmd.Flag.BoolVar(&updateTP, "hmm-update-tp", false, "update HMM transition probabilities, overwrites config file when set")
	cmd.Flag.BoolVar(&updateIP, "hmm-update-ip", false, "update HMM initial probabilities, overwrites config file when set")
	cmd.Flag.StringVar(&outputDistribution, "hmm-output-distribution", DEFAULT_OUT_MODEL, "HMM state output distribution {gaussian, gmm}")
	cmd.Flag.BoolVar(&useAlignments, "use-alignments", false, "train output model using alignments")
	cmd.Flag.StringVar(&tpFilename, "hmm-tp-graph", DEFAULT_TP_FILE, "HMM state transition probabilities graph")
}

func trainer(cmd *Command, args []string) {

	// Read config file.
	fn := fmt.Sprintf("%s%c%s", dir, os.PathSeparator, configFilename)
	data, err := ioutil.ReadFile(fn)
	gjoa.Fatal(err)
	config := gjoa.Config{}
	err = goyaml.Unmarshal(data, &config)
	gjoa.Fatal(err)

	// Default config values.
	if len(config.HMM.OutputDist) == 0 {
		config.HMM.OutputDist = DEFAULT_OUT_MODEL
	}
	if len(config.DataSet) == 0 {
		config.DataSet = DEFAULT_DATA_SET
	}
	if len(config.Model) == 0 {
		config.Model = DEFAULT_MODEL
	}
	if len(config.HMM.TPGraphFilename) == 0 {
		config.HMM.TPGraphFilename = DEFAULT_TP_FILE
	}

	// Overide config when a flag is set.
	// TODO: Is this the best way to do this?
	// I included this to show how it can be done. We probably want to have most
	// params in config and a few in command line to run experiments.
	cmd.Flag.Visit(func(f *flag.Flag) {

		switch f.Name {
		case "model":
			config.Model = modelType
		case "hmm-tp-graph":
			config.HMM.TPGraphFilename = tpFilename
		case "hmm-update-tp":
			config.HMM.UpdateTP = updateTP
		case "hmm-update-ip":
			config.HMM.UpdateIP = updateIP
		case "use-alignments":
			config.HMM.UseAlignments = useAlignments
		case "data-set":
			config.DataSet = dataSet
		case "hmm-output-distribution":
			config.HMM.OutputDist = outputDistribution
		default:
			goto DONE
		}
		glog.Infof("Overwriting config using flag [%s] with value [%v]", f.Name, f.Value)
	DONE:
	})

	// Validations.
	if len(config.DataSet) == 0 {
		glog.Fatalf("DataSet is empty.")
	}

	// Read data set.
	ds, e := gjoa.ReadDataSet(config.DataSet, nil)
	gjoa.Fatal(e)

	// Print config.
	glog.Infof("Read configuration:\n%+v", config)

	// Select model.
	glog.Infof("Training Model: %s.", config.Model)

	// Select the models, here do validation, bookkeeping, etc
	var gs map[string]*gaussian.Gaussian
	switch config.Model {
	case "gaussian":
		gs = trainGaussians(ds)

		if glog.V(1) {
			for k, v := range gs {
				glog.Infof("Model: %s\n%+v", k, v)
			}
		}
	case "gmm":
		glog.Fatalf("Not implemented: %s.", "train gmm")
	case "hmm":
		glog.Infof("Output distribution: %s.", config.HMM.OutputDist)
		graph, tpe := gjoa.ReadFile(config.HMM.TPGraphFilename)
		gjoa.Fatal(tpe)
		nodes, probs := graph.NodesAndProbs()
		glog.V(1).Info(graph.String())

		switch config.HMM.OutputDist {
		case "gaussian":
			if config.HMM.UseAlignments {

				// Trains one Gaussian model per class. Returns a map.
				gs = trainGaussians(ds)

				// Puts Gaussian models in a slice in the same order as probs.
				gaussians := sortGaussians(gs, nodes)

				hmm, e := hmm.NewHMM(probs, nil, gaussians, true, "hmm", &config)
				gjoa.Fatal(e)
				e = hmm.WriteFile(modelOutFilename)
				gjoa.Fatal(e)
			} else {
				glog.Fatalf("Not implemented: %s.", "train fb")
			}

		case "gmm":
			glog.Fatalf("Not implemented: %s.", "output dist gmm")
		default:
			glog.Fatalf("Unknown output distribution: %s.", outputDistribution)
		}
	default:
		glog.Fatalf("Unknown model: %s.", config.Model)
	}
}

func trainGaussians(ds *gjoa.DataSet) (gs map[string]*gaussian.Gaussian) {

	gs = make(map[string]*gaussian.Gaussian)
	var numFrames int
	for {
		features, e := ds.Next()
		if e == io.EOF {
			break
		}
		gjoa.Fatal(e)
		numFrames += len(features)

		for _, v := range features {
			name := v.ClassName
			g, ok := gs[name]
			if !ok {
				g, e = gaussian.NewGaussian(len(v.Values), nil, nil, true, true, v.ClassName)
				gjoa.Fatal(e)
				gs[name] = g
			}
			g.Update(v.Values, 1.0)
		}
	}

	// Estimate params.
	for _, g := range gs {
		g.Estimate()
	}

	return
}

func sortGaussians(gs map[string]*gaussian.Gaussian, nodes []*gjoa.Node) (gaussians []model.Modeler) {

	gaussians = make([]model.Modeler, len(nodes))
	for k, v := range nodes {
		g, found := gs[v.Name]
		if !found {
			glog.Fatalf("There is no model for Node [%s].", v.Name)
		}
		gaussians[k] = g
	}
	return
}
