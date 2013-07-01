package main

import (
	"log"
	"os"

	"bitbucket.org/akualab/gjoa"
	"github.com/gonuts/commander"
	"github.com/gonuts/flag"
)

var cmd *commander.Commander

func init() {
	cmd = &commander.Commander{
		Name: os.Args[0],
		Commands: []*commander.Command{
			gjoa.Recognizer(),
			gjoa.Trainer(),
		},
		Flag: flag.NewFlagSet("gjoa", flag.ExitOnError),
	}
}

func main() {
	err := cmd.Flag.Parse(os.Args[1:])
	if err != nil {
		log.Fatal(err)
	}

	args := cmd.Flag.Args()
	err = cmd.Run(args)
	if err != nil {
		log.Fatal(err)
	}

	return
}
