package gjøa

import (
	"bufio"
	"encoding/json"
	"io"
	"os"
	"testing"
	"time"
)

// Tests

type Eucalyptus struct {
	Abbrev   string
	Rep      float64
	Locality string
	Latitude string
	Altitude float64
	Utility  string
	Date     time.Time
	Tags     string
}

func TestEucalyptus(t *testing.T) {

	f, err := os.Open("data/eucalyptus-data.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	reader := bufio.NewReader(f)

	obj := new(Eucalyptus)
	attr := NewAttribute(reader, obj)

	for {
		euc := new(Eucalyptus)
		err := attr.Next(euc)
		if err == Done {
			t.Log(err)
			return
		} else if err != nil {
			t.Fatal(err)
		}
		t.Logf("eucalyptus: %+v", euc)

	}

}

func TestEucalyptus2(t *testing.T) {

	f, err := os.Open("data/eucalyptus-data.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	reader := bufio.NewReader(f)
	dec := json.NewDecoder(reader)

	for {
		euc := new(Eucalyptus)
		err := dec.Decode(euc)
		if err == io.EOF {
			t.Log(err)
			return
		} else if err != nil {
			t.Fatal(err)
		}
		t.Logf("eucalyptus: %+v", euc)

	}

}
