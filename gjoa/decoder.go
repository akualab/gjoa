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
	glog.Infof("ds:\n%+v", ds)
	// Print config.
	glog.Infof("Read configuration:\n%+v", config)
	// read the hmm from file
	hmm0 := hmm.EmptyHMM()
	glog.Infof("ds:\n%+v", hmm0)
	x, e1 := hmm0.ReadFile(config.HMM.HMMFile)
	if e1 != nil {
		glog.Infof("Problems with ReadFile")
		gjoa.Fatal(e1)
	}
	glog.Infof("modeler:\n%+v", x)
	hmm1 := x.(*hmm.HMM)
	glog.Infof("hmm:\n%+v", hmm1)
	// Use the code in trainer.go trainGaussians(ds *gjoa.DataSet)
	// write a function that given ds and the hmm calls virterbi for each data set
	viterbier(ds, hmm1)
}

func viterbier(ds *gjoa.DataSet, hmm0 *hmm.HMM) {

	for {
		features, e := ds.Next()
		if e == io.EOF {
			break
		}
		gjoa.Fatal(e)

		for _, obs := range features {
			glog.Infof("observations: %v", [][]float64{obs.Values})
			bt, logProbViterbi, err := hmm0.Viterbi([][]float64{obs.Values})
			if err != nil {
				gjoa.Fatal(err)
			}
			glog.Infof("logProbViterbi: %d", logProbViterbi)
			glog.Infof("bt: %s", bt)
		}
	}

}
