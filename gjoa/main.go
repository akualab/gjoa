package main

import (
	"flag"
	"fmt"
	"github.com/golang/glog"
)

var eid string // experiment id
var dir string

func main() {

	commands := NewCommands(
		"gjoa statistical modeling toolkit.",
		cmdTrainer,
		cmdRecognizer,
		cmdGraph,
                cmdDecoder,
	)

	flag.Usage = commands.Usage
	flag.Parse()
	defer glog.Flush()

	args := flag.Args()
	if len(args) < 1 {
		fmt.Printf("Please provide at least one argument try, gjoa -v\n")
		//glog.Fatal(commands.Description)
		return
	}

        // Parsing and running command
	if err := commands.Parse(args); err != nil {
		glog.Fatalf("%s", err)
	}

	glog.Flush()
}
