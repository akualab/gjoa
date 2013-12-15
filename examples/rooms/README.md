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

where `f` is the frame id, `c` is the class name, and `f0, f1` are the features
## Topology

The topology of the house is provided in the file `data/24001/topology-24001.json`. Here are the first few lines:

```YAML
name: southcourt
edges:
 - {from: BACKYARD, to: DINING, weight: 2.0}
 - {from: BACKYARD, to: LIVING, weight: 1.0}
 - {from: BACKYARD, to: KITCHEN, weight: 1.0}
 - {from: BATH1, to: BED1, weight: 3.0}
 - {from: BATH1, to: BED4, weight: 2.0}
```

The name of the house is "southcourt". Each posible transition is indicated in a row. For example,
there is a transition from `BATH1` to `BED1` with a weight of 3.0. The weight indicates how often the
transition is used. These are estimates provided by the home dwellers.

We use a **Gjøa** object of type Graph to create, read, and write grahs as follows:

```Go
    // Read graph.
    g, _ := ReadFile("topology-southcourt.yaml")

    // Write graph.
	g.WriteFile("myhouse.yaml")
```

## Modeling


### The WiFi Corpus

The data is organized by sessions. Each session corresponds to a user (eg. 24001) that collects data using
an Android device in a house. Data from sensors is collected using a fixed sampling period of 4 seconds.
In this case we are collecting the Received Signal Strength Indication (RSSI) from various wireless stationary
base stations located around the home. Each sample is a vector of RSSI measurements in dbm units. Missing
values due to weak signal strength are filled with -100 dbm. In some cases we used heuristics to interpolate
missing values. We also normalized the data to have an exact period of 4 sec. We refer to each sample as a *frame*.

The data is labeled with the name of the room in which the data was collected. The labels were entered manually
in real-time using a simple user-interface on the Android device. Therefore the accuracy of the labels near the
transitions between rooms is not precise. Subjects were asked to set the label at the time they were entering a
room.

### Train One Gaussian per Room

We model the RSSI distribution using a multivariate Gaussian model associated with each room. We name each model with
corresponding room name.

Install gjoa: `go install github.com/akualab/gjoa/gjoa`

Train hmm:

```
gjoa train -use-alignments

# To make it verbose:
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
