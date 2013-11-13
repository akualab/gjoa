// Code to decode an hmm model
package main

import (
	"flag"
	"fmt"
	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model/hmm"
	"github.com/golang/glog"
	"io"
	"io/ioutil"
	"launchpad.net/goyaml"
	"os"
	"path"
)

var cmdDecoder = &Command{
	Run:       decoder,
	UsageLine: "decoder [options]",
	Short:     "runs decoder",
	Long: `
runs decoder.

ex:
 $ gjoa decoder -hmm hmm.json -obs obs.json -out decoding.json
`,
	Flag: *flag.NewFlagSet("gjoa-decoder", flag.ExitOnError),
}

func init() {
	addDecoderFlags(cmdRecognizer)
}

func addDecoderFlags(cmd *Command) {
	defaultDir, err := os.Getwd()
	if err != nil {
		defaultDir = os.TempDir()
	}
	defaultEID := path.Base(defaultDir)
	cmd.Flag.StringVar(&dir, "dir", defaultDir, "the project dir")
	cmd.Flag.StringVar(&eid, "eid", defaultEID, "the experiment id")
}

func decoder(cmd *Command, args []string) {

	fn := fmt.Sprintf("%s%c%s", dir, os.PathSeparator, configFilename)
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
