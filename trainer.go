package gjoa

import (
	"fmt"

	"github.com/gonuts/commander"
	"github.com/gonuts/flag"
)

func Trainer() *commander.Command {
	cmd := &commander.Command{
		Run:       trainer,
		UsageLine: "train [options]",
		Short:     "runs trainer and exits",
		Long: `
runs trainer and exits.

ex:
 $ gjoa train
`,
		Flag: *flag.NewFlagSet("gjoa-train", flag.ExitOnError),
	}
	cmd.Flag.Bool("q", true, "only print error and warning messages, all other output will be suppressed")
	return cmd
}

func trainer(cmd *commander.Command, args []string) {
	name := "gjoa-" + cmd.Name()
	quiet := cmd.Flag.Lookup("q").Value.Get().(bool)
	fmt.Printf("%s: hello from train (quiet=%v)\n", name, quiet)
}
