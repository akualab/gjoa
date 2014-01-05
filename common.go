package gjoa

import "github.com/golang/glog"

type Result struct {
	BatchID string   `json:"batchid"`
	Ref     []string `json:"ref"`
	Hyp     []string `json:"hyp"`
}

func Fatal(err error) {
	if err != nil {
		glog.Fatal(err)
	}
}
