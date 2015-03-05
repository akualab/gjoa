package main

// var graphCommand = cli.Command{
// 	Name:      "graph",
// 	ShortName: "g",
// 	Usage:     "Modifies search graph.",
// 	Description: `runs graph.

// ex:
// $ gjoa graph -in in-graph.json -out out-graph.json
// `,
// 	Action: graphAction,
// 	Flags: []cli.Flag{

// 		cli.StringFlag{"graph-in", "", "the input graph"},
// 		cli.StringFlag{"graph-out", "", "the output graph"},
// 		cli.BoolFlag{"cd-state", "inserts context dependent state for each edge - example: A <=> B becomes A => A-B => B and B => B-A => A"},
// 		cli.Float64Flag{"self-transition", 0, "inserts a self transition arc to each node with the specified weight"},
// 		cli.BoolFlag{"normalize-weights", "converts weights to probabilities before doing any other transformation"},
// 		cli.BoolFlag{"log-weights", "uses log values for arc weights"},
// 	},
// }

// func graphAction(c *cli.Context) {

// 	initApp(c)

// 	if config == nil {
// 		e := fmt.Errorf("Missing config file [%s]", c.String("config-file"))
// 		gjoa.Fatal(e)
// 	}

// 	// Validate parameters. Command flags overwrite config file params.
// 	requiredStringParam(c, "graph-in", &config.HMM.GraphIn)
// 	requiredStringParam(c, "graph-out", &config.HMM.GraphOut)

// 	if c.Bool("cd-state") {
// 		config.HMM.CDState = true
// 	}
// 	if c.Bool("normalize-weights") {
// 		config.HMM.NormalizeWeights = true
// 	}
// 	if c.Bool("log-weights") {
// 		config.HMM.LogWeights = true
// 		glog.Fatalf("log weights not implemented yet")
// 	}

// 	if c.Int("self-transition") > 0 {
// 		config.HMM.SelfTransition = c.Float64("self-transition")
// 	}

// 	if config.HMM.LogWeights {
// 		glog.Fatalf("log weights not implemented yet")
// 	}

// 	if !config.HMM.CDState && config.HMM.SelfTransition <= 0.0 {
// 		glog.Fatalf("No action specified. Exiting.")
// 	}

// 	glog.Infof("Output distribution: %s.", config.HMM.OutputDist)
// 	g, tpe := graph.ReadJSONGraph(config.HMM.GraphIn)

// 	var ng *graph.Graph
// 	var err error

// 	// inserts nodes and self arcs.
// 	if config.HMM.CDState {
// 		ng, err = gjoa.InsertNodes(g, config.HMM.SelfTransition)
// 		if err != nil {
// 			gjoa.Fatal(err)
// 		}
// 	}

// 	// inserts self arcs.
// 	if !config.HMM.CDState && config.HMM.SelfTransition > 0.0 {
// 		ng, err = gjoa.InsertSelfArcs(g, config.HMM.SelfTransition)
// 		if err != nil {
// 			gjoa.Fatal(err)
// 		}
// 	}

// 	if config.HMM.NormalizeWeights {
// 		ng.Normalize(false) // convert weights to probs.
// 		gjoa.Fatal(tpe)
// 	}
// 	nodeNames, probs := ng.TransitionMatrix(false)
// 	glog.V(1).Infof("Graph read:\n%v\n", ng)
// 	glog.V(1).Infof("probs: \n%v\n%v\n", probs, nodeNames)

// 	err = ng.WriteJSONGraph(config.HMM.GraphOut)
// 	if err != nil {
// 		gjoa.Fatal(err)
// 	}

// }
