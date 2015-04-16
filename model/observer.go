package model

import (
	"encoding/json"
	"io"
	"strings"

	"github.com/golang/glog"
)

// Seq is a data format to represent a sequence of observation vectors.
// We use it to read json data.
type Seq struct {
	Vectors    [][]float64 `json:"vectors"`
	Labels     []string    `json:"labels"`
	ID         string      `json:"id"`
	Alignments []*ANode    `json:"alignments,omitempty"`
}

// SeqObserver implements an observer whose undelying values are of type
// FloatObsSequence.
type SeqObserver struct {
	reader io.Reader
}

// NewSeqObserver creates a new SeqObserver. The data is read as a stream of JSON objects
// accessed from an io.Reader. Each JSON object must be separated by a newline.
//
// Example to create an SeqObserver from a file (error handling ignored for brevity).
// The data must be a stream of JSON-encoded Seq values.
//
//   r, _ = os.Open(fn)              // Open file.
//   obs, _ = NewSeqObserver(r)      // Create observer that reads from file.
//   c, _ = obs.ObsChan()            // Get channel. (See model.Observer interface.)
//                                   // Obs type is model.FloatObsSequence.
//  _ = obs.Close()                  // Closes the underlying file reader.
func NewSeqObserver(reader io.Reader) (*SeqObserver, error) {
	so := &SeqObserver{
		reader: reader,
	}
	return so, nil
}

// ObsChan implements the ObsChan method for the observer interface.
// Each observation is a sequence of type model.FloatObsSequence.
func (so *SeqObserver) ObsChan() (<-chan Obs, error) {
	obsChan := make(chan Obs, 1000)
	go func() {

		dec := json.NewDecoder(so.reader)
		for {
			var v Seq
			err := dec.Decode(&v)
			if err == io.EOF {
				break
			}
			if err != nil {
				glog.Warning(err)
				break
			}
			fos := NewFloatObsSequence(v.Vectors, SimpleLabel(strings.Join(v.Labels, ",")), v.ID)
			if v.Alignments != nil && len(v.Alignments) > 0 {
				x := fos.(FloatObsSequence)
				x.SetAlignment(v.Alignments)
			}
		}
		close(obsChan)
	}()
	return obsChan, nil
}

// Close underlying reader if reader implements the io.Closer interface.
func (so *SeqObserver) Close() error {

	c, ok := so.reader.(io.Closer)
	if ok {
		e := c.Close()
		if e != nil {
			return e
		}
	}
	return nil
}
