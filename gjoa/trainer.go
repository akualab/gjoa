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
		cli.IntFlag{"num-iterations, n", 0, "number of training iterations"},

		// Selects a model.
		cli.StringFlag{"model", "", "select a model to train {gaussian, gmm, hmm}"},

		// Single Gaussian flags.

		// Gaussian mixture flags.

		// HMM flags.
		cli.BoolFlag{"update-tp", "update HMM transition probabilities, overwrites config file when set"},
		cli.BoolFlag{"update-ip", "update HMM initial probabilities, overwrites config file when set"},
		cli.StringFlag{"output-distribution", "", "HMM state output distribution {gaussian, gmm}"},
		cli.StringFlag{"align-in", "", "alignment file in"},
		cli.StringFlag{"align-out", "", "alignment file out"},
		cli.StringFlag{"graph-in", "", "HMM state transition probabilities graph"},
		cli.BoolFlag{"expanded-graph", "train using alignments for the expanded graph"},
		cli.BoolFlag{"two-state-collection", "train collection of 2-state hmms. State0: central model, state1: boundary"},
		cli.BoolFlag{"join-collection", "joins a collection of 2-state hmms into a single hmm"},
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
	stringParam(c, "data-set", &config.DataSet)
	requiredStringParam(c, "graph-in", &config.HMM.GraphIn)
	stringParam(c, "align-in", &config.HMM.AlignIn)
	stringParam(c, "align-out", &config.HMM.AlignOut)
	requiredStringParam(c, "model-out", &config.ModelOut)
	stringParam(c, "model-in", &config.ModelIn)

	// If bool flag exists, set param to true overriding config value.
	if c.Bool("update-tp") {
		config.HMM.UpdateTP = true
	}
	if c.Bool("update-ip") {
		config.HMM.UpdateIP = true
	}
	if c.Bool("expanded-graph") {
		config.HMM.ExpandedGraph = true
	}
	if c.Bool("two-state-collection") {
		config.HMM.TwoStateCollection = true
	}
	if c.Bool("join-collection") {
		config.HMM.JoinCollection = true
	}
	intParam(c, "num-iterations", &config.NumIterations)

	// Check vectors
	if len(config.Vectors) == 0 {
		glog.Fatalf("vector specification missing")
	}

	// Read data set.
	var ds *dataframe.DataSet
	var e error
	if len(config.DataSet) > 0 {
		ds, e = dataframe.ReadDataSetFile(config.DataSet)
		gjoa.Fatal(e)
	}

	// Print config.
	glog.Infof("Read configuration:\n%+v", config)

	// Select model.
	glog.Infof("Training Model: %s.", config.Model)

	// Select the models, here do validation, bookkeeping, etc
	var gs map[string]*gaussian.Gaussian
	var models []model.Modeler
	switch config.Model {
	case "gaussian":

		gs = trainGaussians(ds, config.Vectors, nil)

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
		var alignIn map[string]*gjoa.Result
		var alignOut map[string]*gjoa.Result

		switch config.HMM.OutputDist {
		case "gaussian":
			if len(config.HMM.AlignIn) > 0 {
				// Get alignments from file.
				alignIn, e = gjoa.ReadResults(config.HMM.AlignIn)
				gjoa.Fatal(e)
			}

			if config.HMM.JoinCollection {
				hmmColl, e := hmm.ReadHMMCollection(config.ModelIn)
				gjoa.Fatal(e)
				hmmj := hmm.JoinHMMCollection(graph, hmmColl, "joined-hmm")
				e = hmmj.WriteFile(config.ModelOut)
				gjoa.Fatal(e)
				break
			}

			if config.HMM.TwoStateCollection {
				hmm0 := hmm.EmptyHMM()
				hmm1, e1 := hmm0.ReadFile(config.ModelIn) // read hmm file with expanded graph

				gjoa.Fatal(e1)
				hmmColl := initHMMCollection(hmm1.(*hmm.HMM), graph) // original graph

				for i := 0; i < config.NumIterations; i++ {
					glog.Infof("start iteration %d", i)
					for _, h := range hmmColl {
						h.Clear() // Reset stats before training.
					}
					trainHMM(ds, config.Vectors, hmmColl, alignIn)
					glog.V(4).Infof("hmmcoll: \n%+v", hmmColl)
				}
				if e := hmm.WriteHMMCollection(hmmColl, config.ModelOut); e != nil {
					gjoa.Fatal(e)
				}
				goto ALIGN_OUT
			}

			if config.HMM.ExpandedGraph {

				gs, alignOut = trainExpandedGraph(ds, config.Vectors)
			} else {

				gs = trainGaussians(ds, config.Vectors, alignIn)
			}

			// Assigns models to nodes in the graph.
			models = assignGaussians(gs, nodeNames)
			{
				hmm, e := hmm.NewHMM(probs, nil, models, true, "hmm", config)
				gjoa.Fatal(e)

				e = hmm.WriteFile(config.ModelOut)
				gjoa.Fatal(e)
			}

		ALIGN_OUT:
			if len(config.HMM.AlignOut) > 0 {
				if alignOut == nil {
					glog.Warningf("there are no alignments to write")
				}
				gjoa.WriteResults(alignOut, config.HMM.AlignOut)
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

func trainGaussians(ds *dataframe.DataSet, vectors map[string][]string, alignments map[string]*gjoa.Result) (gs map[string]*gaussian.Gaussian) {

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

			var name string
			if len(alignments) == 0 {
				// Get class name using convention.
				// Look up vector named "class".
				name, e = df.String(i, vectors["class"][0])
				gjoa.Fatal(e)
			} else {
				// get alignment from collection.

				r, found := alignments[df.BatchID]
				if !found {
					glog.Fatalf("Can't find alignment for id [%s]. You must provide alignments for the data set.", df.BatchID)
				}
				name = r.Hyp[i]
			}
			// Lookup model for class. Create a new one if not found.
			g, exist := gs[name]
			if !exist {
				g, e = gaussian.NewGaussian(len(feat), nil, nil, true, true, name)
				gjoa.Fatal(e)
				gs[name] = g
				glog.V(1).Infof("Created Gaussian with name %s.", name)
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

func trainExpandedGraph(ds *dataframe.DataSet, vectors map[string][]string) (gs map[string]*gaussian.Gaussian, alignments map[string]*gjoa.Result) {

	alignments = make(map[string]*gjoa.Result)
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

		alignments[df.BatchID] = &gjoa.Result{BatchID: df.BatchID, Hyp: newAlignments}

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
				glog.V(1).Infof("Created Gaussian with name %s.", name)
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

func trainHMM(ds *dataframe.DataSet, vectors map[string][]string, hmms map[string]*hmm.HMM, alignments map[string]*gjoa.Result) {

	var numFrames int
	seq := make([][]float64, 0)
	for {
		df, e := ds.Next() // get next dataframe
		if e == io.EOF {
			break
		}
		gjoa.Fatal(e)
		numFrames += df.N() // add num data instances in dataframe

		last := ""
		for i := 0; i < df.N(); i++ {

			// Get float vector for frame i.
			feat, e := df.Float64Slice(i, vectors["features"]...)

			// Get the label from the data frame or from an alignment file if it exists.
			var name string
			if len(alignments) == 0 {
				// Get class name using convention.
				// Look up vector named "class".
				name, e = df.String(i, vectors["class"][0])
				gjoa.Fatal(e)
			} else {
				// get alignment from collection.

				r, found := alignments[df.BatchID]
				if !found {
					glog.Fatalf("Can't find alignment for id [%s]. You must provide alignments for the data set.", df.BatchID)
				}
				name = r.Hyp[i]
			}

			// Starting a new session
			if len(last) == 0 {
				glog.Warningf("For now we ignore last segment in session. (Improve this later.) len=%d", len(seq))
				seq = seq[:0] // resets slice
			}

			if len(last) > 0 && last != name {
				// update last segment
				key := last + "-" + name
				h, exist := hmms[key]
				if !exist {
					glog.Warningf("Can't find model [%s], possibly bug in training labels, transition not allowed in graph. Skip segment. index: [%d], frame len: [%d], id: [%s]", key, i, df.N(), df.BatchID)
				}
				if len(seq) < 2 {
					glog.Warningf("A training sequence for model %s is too short. Length is %d, skipping.", name, len(seq))
				} else if exist {
					h.Update(seq, 1.0)
				}
				seq = seq[:0] // resets slice
			}
			last = name
			seq = append(seq, feat)
		}
	}

	// Estimate params.
	for _, h := range hmms {
		h.Estimate()
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

// For each node in the graph, create a 2-state HMM with topology:
// -> A -> AB ->
// Where A is a node in the graph for segment A, B is a succesor node, AB is the boundary state
// for transitions from A to B. All states have self transitions.
func initHMMCollection(hmmIn *hmm.HMM, graph *graph.Graph) (hmms map[string]*hmm.HMM) {

	gs := hmmIn.ModelMap()
	config := &gjoa.Config{
		HMM: gjoa.HMM{
			UpdateIP:        false,
			UpdateTP:        false,
			GeneratorSeed:   0,
			GeneratorMaxLen: 100,
		},
	}

	hmms = make(map[string]*hmm.HMM)
	// Iterate over the non-inserted nodes.
	for _, node := range graph.GetAll() {
		isInserted := node.Value().(map[string]interface{})["inserted"].(bool)
		if isInserted {
			continue
		}
		nodeKey := node.Key()
		glog.V(4).Infof("Starting loop node [%s]. ", nodeKey)
		for succ, p := range node.GetSuccesors() {
			succKey := succ.Key()
			if succKey == nodeKey {
				glog.V(4).Infof("Skip transition [%s] -> [%s]. ", nodeKey, succKey)
				continue // skip self transitions.
			}

			isInserted := succ.Value().(map[string]interface{})["inserted"].(bool)
			if !isInserted {
				glog.Fatalf("Succesor is not an inserted node. This shouldn't happen. [%s].", succKey)
			}

			// Gaussians for this HMM
			gaussians := make([]model.Modeler, 2, 2)
			firstG, found1 := gs[nodeKey]
			if !found1 {
				glog.Fatalf("Missing model for node [%s].", nodeKey)
			}
			gaussians[0] = firstG
			secondG, found2 := gs[succKey]
			if !found2 {
				glog.Fatalf("Missing model for node [%s].", succKey)
			}
			gaussians[1] = secondG
			var probs [][]float64
			var initProbs []float64

			glog.Infof("Creating 2-state HMM for nodes [%s] and [%s]. ", nodeKey, succKey)
			probs = [][]float64{{1 - p, p}, {hmm.SMALL_NUMBER, 1 - hmm.SMALL_NUMBER}}
			initProbs = []float64{1 - hmm.SMALL_NUMBER, hmm.SMALL_NUMBER}

			hmm, e := hmm.NewHMM(probs, initProbs, gaussians, true, succKey, config)
			gjoa.Fatal(e)
			hmms[succKey] = hmm
		}
	}
	return
}
