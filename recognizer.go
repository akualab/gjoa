package gjoa

import (
	"fmt"

	"github.com/gonuts/commander"
	"github.com/gonuts/flag"
)

func Recognizer() *commander.Command {
	cmd := &commander.Command{
		Run:       recognizer,
		UsageLine: "rec [options]",
		Short:     "runs recognizer and exits",
		Long: `
runs recognizer and exits.

ex:
 $ gjoa rec
`,
		Flag: *flag.NewFlagSet("gjoa-rec", flag.ExitOnError),
	}
	cmd.Flag.Bool("q", true, "only print error and warning messages, all other output will be suppressed")
	return cmd
}

func recognizer(cmd *commander.Command, args []string) {
	name := "gjoa-" + cmd.Name()
	quiet := cmd.Flag.Lookup("q").Value.Get().(bool)
	fmt.Printf("%s: hello from rec (quiet=%v)\n", name, quiet)
}
