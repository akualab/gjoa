package gjoa

import (
	"encoding/json"
	"io"
	"io/ioutil"
	"launchpad.net/goyaml"
	"os"
)

type DataSet struct {
	Path           string   `yaml:"path"`
	Files          []string `yaml:"files"`
	index          int
	ClassForString map[string]*Class
}

// Reads a list of feature files from a file. See ReadDataSetReader()
func ReadDataSet(fn string, classes map[string]*Class) (ds *DataSet, e error) {

	f, e := os.Open(fn)
	if e != nil {
		return
	}
	ds, e = ReadDataSetReader(f, classes)
	return
}

// Reads a list of feature files from an io.Reader.
// If classes is nil, it will create new unique IDs for each class name.
// Otherwise it will be use the map to lookup the IDs.
// If an class name is missing, it will add it to the map.
// The map is available in struct DataSet field ClassForString.
func ReadDataSetReader(r io.Reader, classes map[string]*Class) (ds *DataSet, e error) {

	var b []byte
	b, e = ioutil.ReadAll(r)
	if e != nil {
		return
	}
	e = goyaml.Unmarshal(b, &ds)
	if e != nil {
		return
	}

	if classes == nil {
		classes = make(map[string]*Class)
	}
	ds.ClassForString = classes
	return
}

// Iterates over files in file list. Returns features.
// The error returns io.EOF when no more files are available.
func (ds *DataSet) Next() (features []Feature, e error) {

	if ds.index == len(ds.Files) {
		return nil, io.EOF
	}
	sep := string(os.PathSeparator)

	features, e = ReadFeaturesFile(ds.Path+sep+ds.Files[ds.index], ds.ClassForString)
	if e != nil {
		return
	}
	ds.index++
	return
}

// Returns array of classes.
func (ds *DataSet) Classes() []Class {

	classes := make([]Class, len(ds.ClassForString))
	var k int
	for _, v := range ds.ClassForString {
		classes[k] = *v
		k++
	}
	return classes
}

type Feature struct {
	Frame     int       `json:"n"`
	ClassName string    `json:"c"`
	ClassID   int       `json:"-"`
	Values    []float64 `json:"data"`
}

type Class struct {
	ID   int
	Freq int64
	Name string
}

// Reads feature from file.
func ReadFeaturesFile(fn string, classes map[string]*Class) (features []Feature, e error) {

	f, e := os.Open(fn)
	if e != nil {
		return
	}
	return ReadFeatures(f, classes)
}

// Reads features from io.Reader.
func ReadFeatures(r io.Reader, classes map[string]*Class) (features []Feature, e error) {

	var b []byte
	b, e = ioutil.ReadAll(r)
	if e != nil {
		return
	}
	e = json.Unmarshal(b, &features)
	if e != nil {
		return nil, e
	}

	// Set unique IDs for labels and update frequencies.
	for k, v := range features {
		// Lookup ID for class name.
		cl, ok := classes[v.ClassName]
		if !ok {
			cl = &Class{ID: len(classes), Freq: 1, Name: v.ClassName}
			classes[v.ClassName] = cl
		} else {
			cl.Freq += 1
		}
		features[k].ClassID = cl.ID
	}
	return
}

func FeaturesForClass(features []Feature, id int) [][]float64 {

	data := make([][]float64, 0)
	for _, v := range features {
		if v.ClassID == id {
			data = append(data, v.Values)
		}
	}
	return data
}
