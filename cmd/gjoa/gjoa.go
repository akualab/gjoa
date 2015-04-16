// Copyright (c) 2015 AKUALAB INC., All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"io/ioutil"
	"os"
	osuser "os/user"
	"path/filepath"
	"runtime"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/akualab/gjoa"
	"github.com/akualab/gjoa/model"
	"github.com/alecthomas/kingpin"
	"github.com/golang/glog"
)

const (
	appName    = "gjoa"
	appVersion = "0.1"
	timeLayout = time.RFC3339
)

var (
	props      *Properties
	logDir     *string
	modelTypes = []string{"Gaussian", "GMM", "HMM"}
)

var (
	app         = kingpin.New(appName, "Gjøa statistical modeling command-line tool. (https://github.com/akualab/gjoa)")
	debug       = app.Flag("debug", "Enable debug mode.").Bool()
	logToStderr = app.Flag("log-stderr", "Logs are written to standard error instead of files.").Default("true").Bool()
	vLevel      = app.Flag("log-level", "Enable V-leveled logging at the specified level.").Default("0").Short('v').String()
	//	modelType   = app.Flag("model", "Model type.").Short('m').Enum(modelTypes...)
	inputModel = app.Flag("input-model", "An input model file.").Short('i').File()
	dataFile   = app.Flag("data", "Data file. See manual for format details.").File()
	dataDir    = app.Flag("dir", "Data dir. See manual for format details.").ExistingDir()
	dim        = app.Flag("dim", "Dimension of the feature vectors.").Int()
	modelName  = app.Flag("model-name", "Name of the model.").String()
	dictFile   = app.Flag("dict", "Dictionary file maps transcription names to model names.").File()

	config    = app.Command("config", "Updates fields in properties file.")
	configMap = config.Arg("properties", "Set properties.").StringMap()

	rand     = app.Command("rand", "Generate random data using model.")
	randSeed = rand.Flag("seed", "Seed for random number generator.").Default("0").Int()

	train   = app.Command("train", "Estimate model parameters.")
	numIter = train.Flag("num-iterations", "Number of training iterations.").Int()

	gaussian = train.Command("gaussian", "Select a Gaussian model.")
	gmm      = train.Command("gmm", "Select a Gaussian mixture model.")
	numGMM   = gmm.Flag("num-components", "Number of GMM components.").Int()

	hmm           = train.Command("hmm", "Select a hidden Markov model.")
	useAlignments = hmm.Flag("use-alignments", "Train from alignments.").Bool()
)

// Properties of gjoa.
type Properties struct {
	Workspace string `toml:"workspace_dir"`
	LogDir    string `toml:"log_dir"`
}

func init() {
	currDir, e1 := os.Getwd()
	gjoa.Fatal(e1)
	propPath := currDir
	u, e2 := osuser.Current()
	if e2 == nil {
		propPath = filepath.Join(u.HomeDir, ".config", "gjoa")
	}
	propPath = filepath.Join(propPath, "properties.toml")
	propEnvVar := os.Getenv("GJOA_PROPERTIES")
	if len(propEnvVar) > 0 {
		propPath = propEnvVar
	}

	// Read toml config file from configPath.
	dat, e3 := ioutil.ReadFile(propPath)
	if e3 == nil {
		_, e4 := toml.Decode(string(dat), config)
		gjoa.Fatal(e4)
	} else {
		props = new(Properties)
		glog.V(2).Infof("unable to read properties file - ", e3)
	}
	defaultLogDir := filepath.Join(currDir, "log")
	if len(props.LogDir) > 0 {
		defaultLogDir = props.LogDir
	}
	logDir = app.Flag("log", "Log output dir.").Default(defaultLogDir).String()
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	app.Version(appVersion)
	cmd := kingpin.MustParse(app.Parse(os.Args[1:]))
	initGlog()
	defer glog.Flush()
	printAppValues()
	checkDir(props.Workspace)
	switch cmd {

	case config.FullCommand():
		glog.V(3).Info("start config command")

	case rand.FullCommand():
		glog.V(3).Info("start rand command")

	case "train gaussian":
		var m model.Trainer
		*numIter = 1
		m = gaussian.NewModel(*dim, gaussian.Name(*modelName))
		doTrain(m)

	case "train gmm":
		var m model.Trainer
		m = gmm.NewModel(*dim, *numGMM, gmm.Name(*modelName))
		doTrain(m)

	case "train hmm":
		var m model.Trainer
		m = hmm.NewModel(hmmOptions()...)
		doTrain(m)

	default:
		app.Usage(os.Stderr)
	}
}

func train(m model.Trainer) {
	obs := getObserver()
	for i := 0; i < *numIter; i++ {
		glog.Infof("iter [%d]", i)
		m.Clear()
		m.Update(obs, model.NoWeight)
		m.Estimate()
	}
	gjoa.WriteJSONFile(fn, m)
}

func hmmOptions() []hmm.Option {
	opt := []hmm.Option{}
	if *dictFile != nil {
		var assigner hmm.MapAssigner
		gjoa.ReadJSON(*dictFile, &assigner)
		glog.Info("reading name mapping file %s", *dictFile)
		glog.V(3).Info("name mapping: %s", assigner)
		opt = append(opt, hmm.OAssign(assigner))
		*dictFile.Close()
	} else {
		var assigner hmm.DirectAssigner
		glog.Info("using direct assigner")
		opt = append(opt, hmm.OAssign(assigner))
	}
	if *useAlignments {
		opt = append(opt, hmm.UseAlignemnts(true))
	}
	return opt
}

func getObserver() model.Observer {
	return nil
}

// Creates dir if it doesn't exist.
func checkDir(path string) {

	if len(path) == 0 {
		return
	}
	e := os.MkdirAll(path, 0755)
	if e != nil {
		glog.Fatal(e)
	}
}

func initGlog() {

	checkDir(*logDir)
	if *logToStderr {
		flag.Set("alsologtostderr", "true")
	}
	flag.Set("v", *vLevel)
	flag.Set("log_dir", *logDir)
}

func printAppValues() {
	glog.Info("app properties:", *props)
	glog.Info("app version: ", appVersion)
	glog.Info("app model type: ", *modelType)
	glog.Info("app log to std err: ", *logToStderr)
	glog.Info("app log level: ", *vLevel)
	glog.Info("app log dir: ", *logDir)
}

func printTrainValues() {
	//	glog.Info("create xxx: ", *someVar)
}
