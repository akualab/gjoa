package main

import (
	"flag"
)

var cmdRecognizer = &Command{
	Run:       recognizer,
	UsageLine: "recognizer [options]",
	Short:     "runs recognizer",
	Long: `
runs recognizer.

ex:
 $ gjoa recognizer
`,
	Flag: *flag.NewFlagSet("gjoa-recognizer", flag.ExitOnError),
}

func init() {
	addRecognizerFlags(cmdRecognizer)
}

func addRecognizerFlags(cmd *Command) {}

func recognizer(cmd *Command, args []string) {}
