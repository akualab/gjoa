package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"

	"github.com/akualab/dataframe"
	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model"
	"github.com/akualab/gjoa/model/gaussian"
	"github.com/akualab/gjoa/model/hmm"
	"github.com/codegangsta/cli"
	"github.com/golang/glog"
	"launchpad.net/goyaml"
)

var trainCommand = cli.Command{
	Name:      "train",
	ShortName: "t",
	Usage:     "Estimates model parameters using data.",
	Description: `runs trainer.

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
	Action: trainAction,
	Flags: []cli.Flag{
		cli.StringFlag{"data-set, d", "", "the file with the list of data files"},
		cli.StringFlag{"model-out, o", "", "output model filename"},
		cli.StringFlag{"model-in, i", "", "input model filename"},

		// Selects a model.
		cli.StringFlag{"model", "", "select a model to train {gaussian, gmm, hmm}"},

		// Single Gaussian flags.

		// Gaussian mixture flags.

		// HMM flags.
		cli.BoolFlag{"update-tp", "update HMM transition probabilities, overwrites config file when set"},
		cli.BoolFlag{"update-ip", "update HMM initial probabilities, overwrites config file when set"},
		cli.StringFlag{"output-distribution", "", "HMM state output distribution {gaussian, gmm}"},
		cli.BoolFlag{"use-alignments", "train output model using alignments"},
		cli.StringFlag{"tp-graph", "", "HMM state transition probabilities graph"},
	},
}

func trainAction(c *cli.Context) {

	initApp(c)

	// Read config file.
	fn := fmt.Sprintf("%s%c%s", dir, os.PathSeparator, configFile)
	data, err := ioutil.ReadFile(fn)
	gjoa.Fatal(err)
	config := gjoa.Config{}
	err = goyaml.Unmarshal(data, &config)
	gjoa.Fatal(err)

	// Validate parameters. Command flags overwrite config file params.
	requiredStringParam(c, "model", &config.Model)
	requiredStringParam(c, "output-distribution", &config.HMM.OutputDist)
	requiredStringParam(c, "data-set", &config.DataSet)
	requiredStringParam(c, "tp-graph", &config.HMM.TPGraphFilename)
	requiredStringParam(c, "model-out", &config.ModelOut)

	// If bool flag exists, set param to true. Ignores config value.
	if c.Bool("update-tp") {
		config.HMM.UpdateTP = true
	}
	if c.Bool("update-ip") {
		config.HMM.UpdateIP = true
	}
	if c.Bool("use-alignments") {
		config.HMM.UseAlignments = true
	}

	// Read data set.
	ds, e := dataframe.ReadDataSet(config.DataSet)
	gjoa.Fatal(e)

	// Print config.
	glog.Infof("Read configuration:\n%+v", config)

	// Select model.
	glog.Infof("Training Model: %s.", config.Model)

	// Select the models, here do validation, bookkeeping, etc
	var gs map[string]*gaussian.Gaussian
	switch config.Model {
	case "gaussian":
		gs = trainGaussians(ds, config.Vectors)

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
				gs = trainGaussians(ds, config.Vectors)

				// Puts Gaussian models in a slice in the same order as probs.
				gaussians := sortGaussians(gs, nodes)

				hmm, e := hmm.NewHMM(probs, nil, gaussians, true, "hmm", &config)
				gjoa.Fatal(e)
				e = hmm.WriteFile(config.ModelOut)
				gjoa.Fatal(e)
			} else {
				glog.Fatalf("Not implemented: %s.", "train forward-backward")
			}

		case "gmm":
			glog.Fatalf("Not implemented: %s.", "output dist gmm")
		default:
			glog.Fatalf("Unknown output distribution: %s.", config.HMM.OutputDist)
		}
	default:
		glog.Fatalf("Unknown model: %s.", config.Model)
	}
}

func trainGaussians(ds *dataframe.DataSet, vectors map[string][]string) (gs map[string]*gaussian.Gaussian) {

	gs = make(map[string]*gaussian.Gaussian)
	var numFrames int
	for {
		df, e := ds.Next() // get next dataframe
		if e == io.EOF {
			break
		}
		gjoa.Fatal(e)
		numFrames += df.N() // add num data instances in dataframe

		//for _, v := range features {
		for i := 0; i < df.N(); i++ {

			// Get float vector for frame i.
			feat, e := df.Float64Slice(i, vectors["features"]...)

			// Get class name using convention.
			// Look up vector named "class".
			name, en := df.String(i, vectors["class"][0])
			gjoa.Fatal(en)

			// Lookup model for class. Create a new one if not found.
			g, exist := gs[name]
			if !exist {
				g, e = gaussian.NewGaussian(len(feat), nil, nil, true, true, name)
				gjoa.Fatal(e)
				gs[name] = g
			}
			g.Update(feat, 1.0)
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
