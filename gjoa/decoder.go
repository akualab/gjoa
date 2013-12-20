// Searches optimal result given the model.
package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"

	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model/hmm"
	"github.com/codegangsta/cli"
	"github.com/golang/glog"
	"launchpad.net/goyaml"
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
	Flags:  []cli.Flag{},
}

func decodeAction(c *cli.Context) {

	initApp(c)

	fn := fmt.Sprintf("%s%c%s", dir, os.PathSeparator, configFile)
	data, err := ioutil.ReadFile(fn)
	gjoa.Fatal(err)
	config := gjoa.Config{}
	err = goyaml.Unmarshal(data, &config)
	gjoa.Fatal(err)
	if len(config.DataSet) == 0 {
		glog.Fatalf("DataSet is empty.")
	}
	// Read data set files
	ds, e := gjoa.ReadDataSet(config.DataSet, nil)
	gjoa.Fatal(e)
	// read the hmm from file
	hmm0 := hmm.EmptyHMM()
	x, e1 := hmm0.ReadFile(config.HMM.HMMFile)
	if e1 != nil {
		glog.Infof("Problems with ReadFile")
		gjoa.Fatal(e1)
	}
	hmm1 := x.(*hmm.HMM)
	viterbier(ds, hmm1)
}

func viterbier(ds *gjoa.DataSet, hmm0 *hmm.HMM) {

	for {
		features, e := ds.Next()
		if e == io.EOF {
			break
		}
		gjoa.Fatal(e)
		// Aggregating the observations
		all := make([][]float64, 0)
		labels := make([]int, 0)
		for _, obs := range features {
			all = append(all, obs.Values)
			labels = append(labels, obs.ClassID)
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
