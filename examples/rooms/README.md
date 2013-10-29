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

Train hmm:

```
gjoa -logtostderr -v=3 train -use-alignments
```

You should see the model trained from Gaussians in `model-out.json`.

Let's use the subcommand `graph` to expand the graph by inserting a new state between the original states.

```
gjoa -logtostderr -v=3 graph -cd-state -in topology.yaml
```

Original:

```YAML
name: random
edges:
  - {from: A, to: A, weight: 1.0}
  - {from: A, to: B, weight: 2.0}
  - {from: A, to: C, weight: 1.0}
  - {from: B, to: B, weight: 1.0}
  - {from: B, to: C, weight: 1.0}
  - {from: C, to: C, weight: 1.0}
  - {from: C, to: A, weight: 3.0}
```

Expanded:

```YAML
name: random CD
edges: [{from: A, to: A, weight: 1}, {from: A, to: A-B, weight: 2}, {from: A-B, to: B,
    weight: 1}, {from: A-B, to: A-B, weight: 1}, {from: A, to: A-C, weight: 1}, {
    from: A-C, to: C, weight: 1}, {from: A-C, to: A-C, weight: 1}, {from: B, to: B,
    weight: 1}, {from: B, to: B-C, weight: 1}, {from: B-C, to: C, weight: 1}, {from: B-C,
    to: B-C, weight: 1}, {from: C, to: C, weight: 1}, {from: C, to: C-A, weight: 3},
  {from: C-A, to: A, weight: 1}, {from: C-A, to: C-A, weight: 1}]
```

## Search


## Results
