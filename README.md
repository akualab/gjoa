# Gjøa Statistical Modeling, Machine Learning, and Pattern Recognition.

## About the Name

[Gjøa](http://en.wikipedia.org/wiki/Gj%C3%B8a) was the first vessel to transit the Northwest Passage. With a crew of six, Roald Amundsen traversed the passage in a three year journey, finishing in 1906.

## Basics

Install: `go install github.com/akualab/gjoa/gjoa` (you specified the full package name under GOPATH) teh command `go run` won't work because the main package uses multiple files.

Print log to stderr. [Learn how to use glog](http://google-glog.googlecode.com/svn/trunk/doc/glog.html)
```
gjoa -logtostderr
```

Run the trainer
```
cd my/project
touch trainer.yaml
gjoa -logtostderr train
```

edit trainer.yaml:
```
model: hmm
hmm:
  update_tp: false
  update_ip: false
  output_distribution: gaussian
```

run:
```
gjoa -logtostderr train
I0930 17:23:34.989853 26920 trainer.go:119] Read configuration:
{Model:hmm HMM:{UpdateTP:false UpdateIP:false OutputDist:gaussian} GMM:{} Gaussian:{}}
I0930 17:23:34.990036 26920 trainer.go:122] Training Model: hmm.
```

overwrite using a command line flag:
```
gjoa -logtostderr train -hmm-update-tp=true
I0930 17:25:14.882555 26930 trainer.go:114] Overwriting config using flag [hmm-update-tp] with value [true]
I0930 17:25:14.882685 26930 trainer.go:119] Read configuration:
{Model:hmm HMM:{UpdateTP:true UpdateIP:false OutputDist:gaussian} GMM:{} Gaussian:{}}
I0930 17:25:14.882719 26930 trainer.go:122] Training Model: hmm.
```

You can see that now `UpdateTP` is `true`. Also notice notice the new line showing what param was overwitten.

General guidelines for config and flags:

* Put most config in the yaml file.
* Attributes that we may want to modify in experiments are good candidates for the comman line so we can set them using a top level script.
