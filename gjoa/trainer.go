package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/akualab/dataframe"
	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model"
	"github.com/akualab/gjoa/model/gaussian"
	"github.com/akualab/gjoa/model/hmm"
	"github.com/akualab/graph"
	"github.com/codegangsta/cli"
	"github.com/golang/glog"
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
		cli.StringFlag{"graph-in", "", "HMM state transition probabilities graph"},
		cli.BoolFlag{"expanded-graph", "train using alignments for the expanded graph"},
	},
}

func trainAction(c *cli.Context) {

	initApp(c)

	if config == nil {
		e := fmt.Errorf("Missing config file [%s]", c.String("config-file"))
		gjoa.Fatal(e)
	}

	// Validate parameters. Command flags overwrite config file params.
	requiredStringParam(c, "model", &config.Model)
	requiredStringParam(c, "output-distribution", &config.HMM.OutputDist)
	requiredStringParam(c, "data-set", &config.DataSet)
	requiredStringParam(c, "graph-in", &config.HMM.GraphIn)
	requiredStringParam(c, "model-out", &config.ModelOut)

	// If bool flag exists, set param to true overriding config value.
	if c.Bool("update-tp") {
		config.HMM.UpdateTP = true
	}
	if c.Bool("update-ip") {
		config.HMM.UpdateIP = true
	}
	if c.Bool("use-alignments") {
		config.HMM.UseAlignments = true
	}
	if c.Bool("expanded-graph") {
		config.HMM.ExpandedGraph = true
	}

	// Read data set.
	ds, e := dataframe.ReadDataSetFile(config.DataSet)
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
		graph, tpe := graph.ReadJSONGraph(config.HMM.GraphIn)
		graph.Normalize(false) // convert weights to probs.
		gjoa.Fatal(tpe)
		nodeNames, probs := graph.TransitionMatrix(false)
		glog.V(1).Infof("Graph read:\n%v\n", graph)

		switch config.HMM.OutputDist {
		case "gaussian":
			if config.HMM.UseAlignments {
				gs = trainGaussians(ds, config.Vectors)
			} else if config.HMM.ExpandedGraph {
				gs = trainExpandedGraph(ds, config.Vectors)
			} else if !config.HMM.UseAlignments && !config.HMM.ExpandedGraph {
				glog.Fatalf("Not implemented: %s.", "train forward-backward")
			}
			// Puts Gaussian models in a slice in the same order as probs.
			gaussians := assignGaussians(gs, nodeNames)

			hmm, e := hmm.NewHMM(probs, nil, gaussians, true, "hmm", config)
			gjoa.Fatal(e)
			e = hmm.WriteFile(config.ModelOut)
			gjoa.Fatal(e)

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

			// Update stats.
			g.Update(feat, 1.0)
		}
	}

	// Estimate params.
	for _, g := range gs {
		g.Estimate()
	}

	return
}

func trainExpandedGraph(ds *dataframe.DataSet, vectors map[string][]string) (gs map[string]*gaussian.Gaussian) {

	var X = 2
	var da []string // for debugging
	gs = make(map[string]*gaussian.Gaussian)
	var numFrames int
	for {
		df, e := ds.Next() // get next dataframe
		if e == io.EOF {
			break
		}
		gjoa.Fatal(e)
		numFrames += df.N() // add num data instances in dataframe

		// First pass: Need to modify alignments as follows. Assign the last X frames
		// of a segment to the inserted node. If segment has fewer than 2X frames don't
		// assign any frames to the new node.
		newAlignments := make([]string, df.N(), df.N())
		if glog.V(3) {
			da = make([]string, df.N(), df.N())
		}
		var nseg int
		lastName := ""
		for i := 0; i < df.N(); i++ {

			// Get class name using convention.
			// Look up vector named "class".
			name, en := df.String(i, vectors["class"][0])
			gjoa.Fatal(en)

			// Modify alignment when we reach the end of a segment.
			if name != lastName {

				if nseg >= 2*X {
					lab := lastName + "-" + name
					for j := 0; j < X; j++ {
						newAlignments[i-j-1] = lab
					}
				}
				lastName = name
				nseg = 0
			}
			newAlignments[i] = name
			nseg += 1
			if glog.V(3) {
				da[i] = name
			}
		}

		if glog.V(3) {
			glog.Infof("Original alignemnt:\n%v\n", da)
			glog.Infof("Modified alignemnt:\n%v\n", newAlignments)
		}

		// Now let's train the gaussians using the new alignments.
		for i := 0; i < df.N(); i++ {

			// Get float vector for frame i.
			feat, e := df.Float64Slice(i, vectors["features"]...)

			// Get class name using the new alignments.
			name := newAlignments[i]

			// Lookup model for class. Create a new one if not found.
			g, exist := gs[name]
			if !exist {
				g, e = gaussian.NewGaussian(len(feat), nil, nil, true, true, name)
				gjoa.Fatal(e)
				gs[name] = g
			}

			// Update stats.
			g.Update(feat, 1.0)
		}
	}

	// Estimate params.
	for _, g := range gs {
		g.Estimate()
	}

	return
}

// Some models may be missing when there are no observatiosn for that model. This function will attempt
// to find the model associated with the previous state and assign a copy.
func assignGaussians(gs map[string]*gaussian.Gaussian, nodes []string) (gaussians []model.Modeler) {

	gaussians = make([]model.Modeler, len(nodes))
	for k, name := range nodes {
		g, found := gs[name]
		if !found {
			glog.Warningf("there is no model for Node [%s]", name)

			// Get predecesor name.
			pName := strings.Split(name, "-")[0]
			pg, found := gs[pName]
			if !found {
				glog.Fatalf("Unable to find predecesor model [%s].", pName)
			}
			glog.Warningf("using predecesior model [%s]", pName)

			// copy model.
			var e error
			g, e = pg.Clone()
			if e != nil {
				glog.Fatal(e)
			}
			g.SetName(name)
		}
		gaussians[k] = g
	}
	return
}
