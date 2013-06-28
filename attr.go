package gjøa

import (
	"encoding/json"
	"errors"
	"io"
	"reflect"
)

var Done = errors.New("Attribute: there are no more data instances.")

// The attributes of a dataset.
type Attribute struct {
	Name    string
	obj     interface{}
	reader  io.Reader
	decoder *json.Decoder
}

// Creates a new dataset of the same type as obj. Streams data from reader.
func NewAttribute(reader io.Reader, obj interface{}) (attr *Attribute) {

	attr = &Attribute{

		// Get the type name.
		Name: reflect.TypeOf(obj).Elem().Name(),

		// An instance of the object type. We use it for reflection.
		obj: obj,

		// Initialize the JSON decoder.
		decoder: json.NewDecoder(reader),
	}
	return
}

// Get the next instance of the data. Returns attr.Done when no more
// data is available.
func (attr *Attribute) Next(v interface{}) (err error) {

	err = attr.decoder.Decode(v)
	if err == io.EOF {
		return Done
	}
	return
}
