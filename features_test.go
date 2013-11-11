package gjoa

import (
	"io"
	"io/ioutil"
	"os"
	"testing"
)

func TestDataSet(t *testing.T) {

	// Prepare dirs.
	tmpDir := os.TempDir()
	os.MkdirAll(tmpDir+"data", 0755)

	// Create file list yaml file.
	fn := tmpDir + "filelist.yaml"
	t.Logf("File List: %s.", fn)
	err := ioutil.WriteFile(fn, []byte(filelistData), 0644)
	CheckError(t, err)

	// Create data file 1.
	f1 := tmpDir + "data" + string(os.PathSeparator) + "file1.json"
	t.Logf("Data File 1: %s.", f1)
	e := ioutil.WriteFile(f1, []byte(file1), 0644)
	CheckError(t, e)

	// Create data file 2.
	f2 := tmpDir + "data" + string(os.PathSeparator) + "file2.json"
	t.Logf("Data File 2: %s.", f2)
	e = ioutil.WriteFile(f2, []byte(file2), 0644)
	CheckError(t, e)

	// Read file list.
	fl, e := ReadDataSet(fn, nil)
	CheckError(t, e)

	// Check DataSet content.
	t.Logf("DataSet: %v", fl)

	if fl.Path != "data" {
		t.Fatalf("Path is [%s]. Expected \"data\".", fl.Path)
	}
	if fl.Files[0] != "file1.json" {
		t.Fatalf("Files[0] is [%s]. Expected \"file1\".", fl.Files[0])
	}
	if fl.Files[1] != "file2.json" {
		t.Fatalf("Files[1] is [%s]. Expected \"file2\".", fl.Files[1])
	}

	os.Chdir(tmpDir)
	var n int
	for {
		features, e := fl.Next()
		if e == io.EOF {
			break
		}
		CheckError(t, e)
		t.Logf("data: \n%v\n", features)
		n += len(features)
	}

	if n != 12 {
		t.Fatalf("Num features is [%d]. Expected \"12\".", n)
	}

	nClasses := len(fl.ClassForString)
	if nClasses != 3 {
		t.Fatalf("Num classes is [%d]. Expected \"3\".", nClasses)
	}

}

const filelistData string = `
path: data
files:
  - file1.json
  - file2.json
`
const file1 string = `[
{"n":0,"c":"BED5","data":[-40.8,-41.2]},
{"n":1,"c":"BED5","data":[-41.8,-41.1]},
{"n":2,"c":"BED5","data":[-42.8,-40.34]},
{"n":3,"c":"DINING","data":[-42.9,-40.11]},
{"n":4,"c":"DINING","data":[-42.764,-39.98]},
{"n":5,"c":"DINING","data":[-42.209,-39.6]}]
`
const file2 string = `[
{"n":0,"c":"KITCHEN","data":[-20.1,-31.3]},
{"n":1,"c":"KITCHEN","data":[-21.8,-31.1]},
{"n":2,"c":"KITCHEN","data":[-22.8,-30.21]},
{"n":3,"c":"DINING","data":[-22.9,-30.99]},
{"n":4,"c":"DINING","data":[-22.22,-29.76]},
{"n":5,"c":"DINING","data":[-22.345,-29.6]}]
`
