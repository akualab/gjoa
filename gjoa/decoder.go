// Code to decode an hmm model
package main

import (
        "flag"
        "fmt"
        "github.com/akualab/gjoa"
        "github.com/golang/glog"
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
        glog.Info("config %v", config)
        if len(config.DataSet) == 0 {
                glog.Fatalf("DataSet is empty.")
        }
        // Read data set.
        ds, e := gjoa.ReadDataSet(config.DataSet, nil)
        gjoa.Fatal(e)
        glog.Info("ds %v", ds)
}
