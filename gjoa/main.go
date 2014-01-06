package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path"
	"path/filepath"

	"github.com/akualab/gjoa"
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
	config     *gjoa.Config
)

func main() {

	// Set a defaults.
	defaultDir, err := os.Getwd()
	if err != nil {
		defaultDir = os.TempDir()
	}
	defaultEid := path.Base(defaultDir)
	defaultLogDir := defaultDir + string(os.PathSeparator) + "logs"

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
		cli.StringFlag{"log-dir, l", defaultLogDir, "log directory"},
	}

	app.Commands = []cli.Command{
		trainCommand,
		graphCommand,
		decodeCommand,
		scoreCommand,
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

	// Also log to std error.
	flag.Set("alsologtostderr", "true")

	// Set glog debug level
	flag.Set("v", fmt.Sprintf("%d", debugLevel))

	// Set logs dir.
	logsDir := c.GlobalString("log-dir")
	flag.Set("log_dir", logsDir)
	if e := os.MkdirAll(logsDir, 0755); e != nil {
		gjoa.Fatal(e)
	}

	// Change working directory.
	if e := os.Chdir(dir); e != nil {
		gjoa.Fatal(e)
	}
	glog.Infof("working dir: %s", dir)

	// Read config file.
	fn, e := filepath.Abs(configFile)
	if e != nil {
		gjoa.Fatal(e)
	}
	glog.Infof("config file: %s", fn)

	// If config file exists, read it.
	// Commands must check if a config is needed.
	exist, e := exists(fn)
	gjoa.Fatal(e)
	if exist {
		config, e = gjoa.ReadConfig(fn)
		gjoa.Fatal(e)
	}
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

// Missing config value.
var NoConfigValueError = errors.New("config value not found")

// Uses command line flag if present. Otherwise uses config file param.
// Returns NoConfigValueError if no value is specified.
func stringParam(c *cli.Context, flag string, configParam *string) error {

	flagValue := c.String(flag)

	// Validate parameter. Command flags overwrite config file params.
	if len(flagValue) > 0 {

		glog.Infof("Overwriting config using flag [%s] with value [%v]", flag, flagValue)

		// Use command flag, ignore config file.
		*configParam = flagValue
		return nil

	}

	if len(*configParam) == 0 {

		// Value missing.
		return NoConfigValueError
	}

	return nil
}

// exists returns whether the given file or directory exists or not
func exists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}
