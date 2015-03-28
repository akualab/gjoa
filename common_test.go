package gjoa

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWrite(t *testing.T) {

	x := []float64{1.1, 2.2, 3.3}
	var y []float64

	fn := filepath.Join(os.TempDir(), "floats.json")
	WriteJSONFile(fn, x)
	t.Logf("Wrote to temp file: %s\n", fn)

	// Read back.
	e := ReadJSONFile(fn, &y)
	if e != nil {
		t.Fatal(e)
	}

	t.Logf("Original:%v", x)
	t.Logf("Read back from file:%v", y)

	for k, v := range x {
		if v != y[k] {
			t.Fatal("write/read mismatched")
		}
	}
}
