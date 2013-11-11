package main

import (
	"flag"
	"github.com/akualab/gjoa"
	"github.com/golang/glog"
	"os"
	"path"
)

var cmdGraph = &Command{
	Run:       graph,
	UsageLine: "graph [options]",
	Short:     "runs graph",
	Long: `
runs graph.

ex:
 $ gjoa graph -in in-graph.json -out out-graph.json
`,
	Flag: *flag.NewFlagSet("gjoa-graph", flag.ExitOnError),
}

var contextDependant bool
var inFilename, outFilename string

func init() {
	addGraphFlags(cmdGraph)
}

func addGraphFlags(cmd *Command) {

	defaultDir, err := os.Getwd()
	if err != nil {
		defaultDir = os.TempDir()
	}
	defaultEID := path.Base(defaultDir)

	cmd.Flag.StringVar(&dir, "dir", defaultDir, "the project dir")
	cmd.Flag.StringVar(&eid, "eid", defaultEID, "the experiment id")
	cmd.Flag.StringVar(&inFilename, "in", "graph-in.yaml", "the input graph")
	cmd.Flag.StringVar(&outFilename, "out", "graph-out.yaml", "the output graph")
	cmd.Flag.BoolVar(&contextDependant, "cd-state", false, "inserts context dependent state for each edge - example: A <=> B becomes A => A-B => B and B => B-A => A")

}

func graph(cmd *Command, args []string) {

	if !contextDependant {
		// For now there is nothig else to do.
		glog.Fatalf("Nothing to do.")
	}

	if contextDependant {
		g, tpe := gjoa.ReadFile(inFilename)
		gjoa.Fatal(tpe)
		nodes, probs := g.NodesAndProbs()

		glog.V(1).Info("Input Graph:\n")
		glog.V(1).Info(g.String())

		glog.Infof("probs: \n%v\n%v\n", probs, nodes)

		ng, cdNodes := g.InsertContextDependentStates()

		for k, v := range cdNodes {
			glog.Infof("CD Node: [%s] %v", k, v)
		}
		ng.WriteFile(outFilename)
	}
}
