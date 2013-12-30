// Searches optimal result given the model.
package main

import (
	"fmt"
	"io"

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
	viterbier(ds, config.Vectors, hmm1)
}

func viterbier(ds *dataframe.DataSet, vectors map[string][]string, hmm0 *hmm.HMM) {

	for {
		df, e := ds.Next()
		if e == io.EOF {
			break
		}
		gjoa.Fatal(e)

		// Aggregating the observations
		all := make([][]float64, 0)
		labels := make([]int, 0)
		nameToIndex := hmm0.Indices()
		for i := 0; i < df.N(); i++ {

			// Get float vector for frame i.
			feat, e := df.Float64Slice(i, vectors["features"]...)
			gjoa.Fatal(e)

			// Get class name using convention.
			// Look up vector named "class".
			name, en := df.String(i, vectors["class"][0])
			gjoa.Fatal(en)
			idx := nameToIndex[name]
			all = append(all, feat)
			labels = append(labels, idx)
		}

		// Running viterbi and metric for one
		// sequence of observations
		bt, _, err := hmm0.Viterbi(all)
		if err != nil {
			gjoa.Fatal(err)
		}
		glog.Infof("labels: %+v", labels)
		glog.Infof("viterb: %+v", bt)
	}
}
