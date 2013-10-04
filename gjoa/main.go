package main

import (
	"flag"
	"fmt"
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
		fmt.Printf("Please provide at least one argument try \ngjoa -v")
		//glog.Fatal(commands.Description)
		return
	}

	if err := commands.Parse(args); err != nil {
		glog.Fatalf("%s", err)
	}
}
