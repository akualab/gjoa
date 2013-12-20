package main

import (
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

		cli.StringFlag{"in", "", "the input graph"},
		cli.StringFlag{"out", "", "the output graph"},
		cli.BoolFlag{"cd-state", "inserts context dependent state for each edge - example: A <=> B becomes A => A-B => B and B => B-A => A"},
	},
}

func graphAction(c *cli.Context) {

	initApp(c)

	contextDependant := c.Bool("cd-state")
	if !contextDependant {
		// For now there is nothig else to do.
		glog.Fatalf("Nothing to do.")
	}

	if contextDependant {
		g, tpe := gjoa.ReadFile(c.String("in"))
		gjoa.Fatal(tpe)
		nodes, probs := g.NodesAndProbs()

		glog.V(1).Info("Input Graph:\n")
		glog.V(1).Info(g.String())

		glog.Infof("probs: \n%v\n%v\n", probs, nodes)

		ng, cdNodes := g.InsertContextDependentStates()

		for k, v := range cdNodes {
			glog.Infof("CD Node: [%s] %v", k, v)
		}
		ng.WriteFile(c.String("out"))
	}
}
