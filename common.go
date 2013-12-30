package gjoa

import "github.com/golang/glog"

func Fatal(err error) {
	if err != nil {
		glog.Fatal(err)
	}
}
