package main

import (
	"fmt"

	"github.com/akualab/gjoa"
	"github.com/codegangsta/cli"
	"github.com/golang/glog"
)

var graphCommand = cli.Command{
	Name:      "graph",
	ShortName: "g",
	Usage:     "Modifies search graph.",
	Description: `runs graph.

ex:
$ gjoa graph -in in-graph.json -out out-graph.json
`,
	Action: graphAction,
	Flags: []cli.Flag{

		cli.StringFlag{"graph-in", "", "the input graph"},
		cli.StringFlag{"graph-out", "", "the output graph"},
		cli.BoolFlag{"cd-state", "inserts context dependent state for each edge - example: A <=> B becomes A => A-B => B and B => B-A => A"},
	},
}

func graphAction(c *cli.Context) {

	initApp(c)

	if config == nil {
		e := fmt.Errorf("Missing config file [%s]", c.String("config-file"))
		gjoa.Fatal(e)
	}

	// Validate parameters. Command flags overwrite config file params.
	requiredStringParam(c, "graph-in", &config.HMM.GraphIn)
	requiredStringParam(c, "graph-out", &config.HMM.GraphOut)

	if c.Bool("cd-state") {
		config.HMM.CDState = true
	}
	if !config.HMM.CDState {
		// For now there is nothig else to do.
		glog.Fatalf("For now, only cd-state=true is supported.")
	}

	if config.HMM.CDState {
		g, tpe := gjoa.ReadFile(config.HMM.GraphIn)
		glog.Errorf("error trying to open graph file [%s]", config.HMM.GraphIn)
		gjoa.Fatal(tpe)
		nodes, probs := g.NodesAndProbs()

		glog.V(1).Info("Input Graph:\n")
		glog.V(1).Info(g.String())

		glog.Infof("probs: \n%v\n%v\n", probs, nodes)

		ng, cdNodes := g.InsertContextDependentStates()

		for k, v := range cdNodes {
			glog.Infof("CD Node: [%s] %v", k, v)
		}
		ng.WriteFile(config.HMM.GraphOut)
	}
}
