// Searches optimal result given the model.
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/akualab/gjoa"
	"github.com/akualab/scorer"
	"github.com/codegangsta/cli"
	"github.com/golang/glog"
)

var scoreCommand = cli.Command{
	Name:      "score",
	ShortName: "d",
	Usage:     "Compute various metrics for results file.",
	Description: `
runs scorer.

ex:
$ gjoa score
`,
	Action: scoreAction,
	Flags: []cli.Flag{
		cli.StringFlag{"results-file, r", "", "the input file with results to be analyzed"},
		cli.StringFlag{"score-file, t", "", "output score file"},
		cli.BoolFlag{"split-dash", "maps the hyp token using strings.Split(token, \"-\")[0]"},
	},
}

func scoreAction(c *cli.Context) {

	initApp(c)

	if config == nil {
		e := fmt.Errorf("Missing config file [%s]", c.String("config-file"))
		gjoa.Fatal(e)
	}

	// Validate parameters. Command flags overwrite config file params.
	requiredStringParam(c, "results-file", &config.ResultsFile)

	var scoreFile *os.File
	if e := stringParam(c, "score-file", &config.ScoreFile); e == NoConfigValueError {
		glog.Infof("no score file specified, writing to stdout")
		scoreFile = os.Stdout
	} else {
		var err error
		scoreFile, err = os.Create(config.ScoreFile)
		gjoa.Fatal(err)
		defer scoreFile.Close()
	}

	// If bool flag exists, set param to true overriding config value.
	if c.Bool("split-dash") {
		config.HMM.SplitDash = true
	}

	// Open results file to start reading.
	resultsFile, e := os.Open(config.ResultsFile)
	if e != nil {
		gjoa.Fatal(e)
	}
	defer resultsFile.Close()

	reader := bufio.NewReader(resultsFile)
	sc := scorer.NewAccuracyScore(true, false)
	for {
		b, eb := reader.ReadBytes('\n')
		if eb == io.EOF {
			break
		}
		if eb != nil {
			gjoa.Fatal(eb)
		}

		result := new(gjoa.Result)
		if e := json.Unmarshal(b, result); e != nil {
			gjoa.Fatal(e)
		}

		var hyp []string
		var ref []string
		if config.HMM.SplitDash {
			hyp = splitDash(result.Hyp)
		} else {
			hyp = result.Hyp
		}
		ref = result.Ref
		score, e := sc.Session(result.BatchID, ref, hyp)
		if e != nil {
			gjoa.Fatal(e)
		}
		fmt.Fprintf(scoreFile, "ID: %s, acc: %s\n", result.BatchID, score.Text)
		glog.V(3).Infof("\nREF: %v", ref)
		glog.V(3).Infof("\nHYP: %v", hyp)
	}

	score := sc.Total()
	if e != nil {
		gjoa.Fatal(e)
	}
	fmt.Fprintf(scoreFile, "\nAVG: acc: %s\n", score.Text)
}

func splitDash(in []string) (out []string) {

	out = make([]string, len(in))
	for k, v := range in {
		if strings.Index(v, "-") == -1 {
			out[k] = v // no dash
		} else {
			t := strings.Split(v, "-")[0]
			out[k] = t // substring before dash
		}
	}
	return
}
