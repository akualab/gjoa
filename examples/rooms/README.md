# Room Recogition using Statistical Models

The task is to recognize the trajectory of a mobile phone inside a house. The house is divided in "rooms"
and the topology of the rooms is known. The transitions within the home are constrained by the topology.
The recognizer must be able to hypothesize the most likely sequence of rooms using features recorded by
the mobile phone.

## Data

The data is stored in files using the following format:

```JSON
[
    {"n":0,"c":"BED5","data":[-40.8,-41.2]},
    {"n":1,"c":"BED5","data":[-41.8,-41.1]},
    {"n":2,"c":"BED5","data":[-42.8,-40.34]},
    {"n":3,"c":"DINING","data":[-42.9,-40.11]},
    {"n":4,"c":"DINING","data":[-42.764,-39.98]},
    {"n":5,"c":"DINING","data":[-42.209,-39.6]}
]
```

where `f` is the frema id, `c` is the class name, and `f0, f1` are the features
## Topology

The topology of the house is provided in the file topology-southcourt.yaml. Here is a sample:

```YAML
name: southcourt
edges:
  - {from: BACKYARD, to: DINING, weight: 2.0}
  - {from: BACKYARD, to: LIVING, weight: 1.0}
  - {from: BACKYARD, to: KITCHEN, weight: 1.0}
  - {from: BATH1, to: BED1, weight: 3.0}
  - {from: BATH1, to: BED2, weight: 2.0}
  - {from: BATH2, to: BED4, weight: 2.0}
  - {from: BATH2, to: BED2, weight: 2.0}
  - {from: BATH3, to: BED5, weight: 2.0}
  - {from: BATH3, to: DINING, weight: 3.0}
  - {from: BED1, to: BED4, weight: 2.0}
  - {from: BED1, to: BATH1, weight: 2.0}
```

The name of the house is "southcourt". Each posible transition is indicated in a row. For example,
there is a transition from `BATH1` to `BED2` with a weight of 2.0. The weight indicates how often the
transition is used. These are estimates provided by the home dwellers.

**Gjøa** provides a Graph object to create, read, and write grahs as follows:

```Go
    // Read graph.
    g, _ := ReadFile("topology-southcourt.yaml")

    // Write graph.
	g.WriteFile("myhouse.yaml")
```

## Modeling

We have a sequence of feature vectors sampled every 4 seconds. The simplest approach is to train a
multivariate Gaussian model for each room using the labeled data. In the next section we will show how to
search the optimal room sequence using a Markov model.

### Train Gaussians

   Remember that if you change the call you need to run `go install` in project main under `ROOT/gjoa`.

Train a bunch of gaussians:

```
gjoa -logtostderr -v=3 train
```

The output lokks like this:

```
$ gjoa -logtostderr -v=3 train
I1015 22:29:21.373001 67853 trainer.go:120] Read configuration:
{DataSet:train.yaml Model:gaussian HMM:{UpdateTP:false UpdateIP:false OutputDist: GeneratorSeed:0 GeneratorMaxLen:0} GMM:{} Gaussian:{}}
I1015 22:29:21.373158 67853 trainer.go:123] Training Model: gaussian.
I1015 22:29:21.373169 67853 features.go:62] feature file: data/train/southcourt-001.json
I1015 22:29:21.373391 67853 features.go:62] feature file: data/train/southcourt-002.json
I1015 22:29:21.373527 67853 trainer.go:132] Model: BED5
&{BaseModel:0xc200093880 ModelName:BED5 NE:2 IsTrainable:true NSamples:3 Diag:true Sumx:[-125.39999999999999 -122.64000000000001] Sumxsq:[5243.719999999999 5013.965600000001] Mean:[-41.8 -40.88] StdDev:[0.8164965809276332 0.3840138886381775] variance:[0.6666666666665151 0.14746666666701458] varianceInv:[1.500000000000341 6.781193490038251] tmpArray:[-0.40546510810839176 -1.9141531174397624] const1:-1.8378770664093453 const2:-0.6780679536352683 fpool:0xc200089980}
I1015 22:29:21.373637 67853 trainer.go:132] Model: DINING
&{BaseModel:0xc2000938a0 ModelName:DINING NE:2 IsTrainable:true NSamples:6 Diag:true Sumx:[-195.338 -210.04] Sumxsq:[6968.206802000001 7497.5702] Mean:[-32.55633333333333 -35.00666666666666] StdDev:[10.072386024285537 4.91205885777263] variance:[101.4529602222226 24.12832222222255] varianceInv:[0.009856784837126483 0.04144506985566468] tmpArray:[4.619595244971036 3.1833863464372407] const1:-1.8378770664093453 const2:-5.739367862113483 fpool:0xc200089a50}
I1015 22:29:21.373704 67853 trainer.go:132] Model: KITCHEN
&{BaseModel:0xc200093b40 ModelName:KITCHEN NE:2 IsTrainable:true NSamples:3 Diag:true Sumx:[-64.7 -92.61000000000001] Sumxsq:[1399.0900000000001 2859.5441] Mean:[-21.566666666666666 -30.870000000000005] StdDev:[1.114550233153386 0.47377913278915557] variance:[1.2422222222222672 0.22446666666644433] varianceInv:[0.8050089445437991 4.455004455008868] tmpArray:[0.21690189039076993 -1.494028060924263] const1:-1.8378770664093453 const2:-1.1993139811425988 fpool:0xc200089b40}
```

## Search


## Results
