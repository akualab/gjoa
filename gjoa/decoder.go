// Searches optimal result given the model.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/akualab/dataframe"
	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model/hmm"
	"github.com/codegangsta/cli"
	"github.com/golang/glog"
)

var decodeCommand = cli.Command{
	Name:      "decode",
	ShortName: "d",
	Usage:     "Searches optimal result given the model.",
	Description: `
runs decoder.

ex:
$ gjoa decode
`,
	Action: decodeAction,
	Flags: []cli.Flag{
		cli.StringFlag{"data-set, d", "", "the file with the list of data files"},
		cli.StringFlag{"hmm-file, f", "", "input hmm file"},
		cli.StringFlag{"results-file, r", "", "results file"},
	},
}

func decodeAction(c *cli.Context) {

	initApp(c)

	if config == nil {
		e := fmt.Errorf("Missing config file [%s]", c.String("config-file"))
		gjoa.Fatal(e)
	}

	// Validate parameters. Command flags overwrite config file params.
	requiredStringParam(c, "data-set", &config.DataSet)
	requiredStringParam(c, "hmm-file", &config.HMM.HMMFile)

	var resultsFile *os.File
	if e := stringParam(c, "results-file", &config.ResultsFile); e == NoConfigValueError {
		glog.Infof("no results file specified, writing to stdout")
		resultsFile = os.Stdout
	} else {

		// Open results file.
		var err error
		resultsFile, err = os.Create(config.ResultsFile)
		gjoa.Fatal(err)
		defer resultsFile.Close()
	}

	// Read data set files
	ds, e := dataframe.ReadDataSetFile(config.DataSet)
	gjoa.Fatal(e)

	// Read hmm from file.
	hmm0 := hmm.EmptyHMM()
	x, e1 := hmm0.ReadFile(config.HMM.HMMFile)
	if e1 != nil {
		glog.Errorln("Problems with ReadFile")
		gjoa.Fatal(e1)
	}
	hmm1 := x.(*hmm.HMM)
	glog.V(4).Infof("HMM passed to viterbier:\n%+v", hmm1)
	viterbier(ds, config.Vectors, hmm1, resultsFile)
}

func viterbier(ds *dataframe.DataSet, vectors map[string][]string, hmm0 *hmm.HMM, w io.Writer) {

	// Prepare JSON decoder.
	enc := json.NewEncoder(w)

	features := make([][]float64, 0)
	labels := make([]int, 0)
	refs := make([]string, 0)
	hyps := make([]string, 0)
	nameToIndex := hmm0.Indices()
	invIndex := make([]string, len(nameToIndex))
	glog.Infof("Starting decoder with %d models.", len(nameToIndex))
	// Get inverse index to recover hyp model names int -> string
	for k, v := range nameToIndex {
		invIndex[v] = k
	}

	for {
		df, e := ds.Next()
		if e == io.EOF {
			break
		}
		gjoa.Fatal(e)

		// Reset slices.
		features = features[:0]
		labels = labels[:0]
		refs = refs[:0]
		hyps = hyps[:0]
		id := df.BatchID

		for i := 0; i < df.N(); i++ {

			// Get float vector for frame i.
			feat, e := df.Float64Slice(i, vectors["features"]...)
			gjoa.Fatal(e)

			// Get class name using convention.
			// Look up vector named "class".
			name, en := df.String(i, vectors["class"][0])
			gjoa.Fatal(en)
			idx := nameToIndex[name]
			features = append(features, feat)
			labels = append(labels, idx)
			refs = append(refs, name)
		}

		// Running viterbi and metric for one
		// sequence of observations
		bt, _, err := hmm0.Viterbi(features)
		if err != nil {
			gjoa.Fatal(err)
		}
		// glog.Infof("ref: %+v", labels)
		// glog.Infof("hyp: %+v", bt)

		// convert bt indices to names
		for _, v := range bt {
			hyps = append(hyps, invIndex[v])
		}

		f, ok := w.(*os.File)
		if ok && f.Name() == "/dev/stdout" {
			// Write without encoding.
			glog.Infof("id: %s, ref: %v", id, strings.Join(refs, " "))
			glog.Infof("id: %s, hyp: %v", id, strings.Join(hyps, " "))
		} else {
			// Use json encoding.
			result := gjoa.Result{BatchID: id, Ref: refs, Hyp: hyps}
			enc.Encode(result)
		}
	}

}
