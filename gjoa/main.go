package main

import (
	"flag"
	"github.com/golang/glog"
)

func main() {

	commands := NewCommands(
		"gjoa statistical modeling toolkit.",
		cmdTrainer,
		cmdRecognizer,
	)

	flag.Usage = commands.Usage
	flag.Parse()
	defer glog.Flush()

	args := flag.Args()
	if len(args) < 1 {
		glog.Fatal(commands.Description)
	}

	if err := commands.Parse(args); err != nil {
		glog.Fatalf("%s", err)
	}
}
