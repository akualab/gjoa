package main

import (
	"flag"
	"fmt"
	"os"
	"path"

	"github.com/codegangsta/cli"
	"github.com/golang/glog"
)

// Global constants.
const (
	DEFAULT_CONFIG_FILE = "gjoa.yaml"
)

// Global Vars
var (
	configFile string
	eid        string
	dir        string
)

func main() {

	// Set a default directory and id for the experiment.
	defaultDir, err := os.Getwd()
	if err != nil {
		defaultDir = os.TempDir()
	}
	defaultEid := path.Base(defaultDir)

	app := cli.NewApp()
	app.Name = "gjoa"
	app.Usage = "Statistical Modeling Toolkit"
	app.Version = Version
	app.Email = "info@akualab.com"
	app.Action = func(c *cli.Context) {
		cli.ShowAppHelp(c)
	}

	app.Flags = []cli.Flag{
		cli.StringFlag{"dir, d", defaultDir, "experiment directory"},
		cli.StringFlag{"eid, e", defaultEid, "experiment id"},
		cli.StringFlag{"config-file, c", DEFAULT_CONFIG_FILE, "configuration file"},
		cli.IntFlag{"debug, g", 0, "verbose level"},
	}

	app.Commands = []cli.Command{
		trainCommand,
		graphCommand,
		decodeCommand,
	}

	app.Run(os.Args)

	glog.Flush()
}

// Common for all apps.
func initApp(c *cli.Context) {

	configFile = c.GlobalString("config-file")
	dir = c.GlobalString("dir")
	eid = c.GlobalString("eid")

	debugLevel := c.GlobalInt("debug")

	// Log to std error.
	flag.Set("logtostderr", "true")

	// Set glog debug level
	flag.Set("v", fmt.Sprintf("%d", debugLevel))

	// TODO add options to log to dir, etc.
}

// Uses command line flag if present. Otherwise uses config file param.
// Fatal error if config file param is also missing.
func requiredStringParam(c *cli.Context, flag string, configParam *string) {

	flagValue := c.String(flag)

	// Validate parameter. Command flags overwrite config file params.
	if len(flagValue) > 0 {

		glog.Infof("Overwriting config using flag [%s] with value [%v]", flag, flagValue)

		// Use command flag, ignore config file.
		*configParam = flagValue
		return

	}

	if len(*configParam) == 0 {

		// Value missing.
		glog.Fatalf("missing parameter: [%s]", flag)
	}
}
