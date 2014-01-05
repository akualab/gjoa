# Room Recogition using Statistical Models

The task is to recognize the trajectory of a mobile phone inside a house. The house is divided in "rooms"
and the topology of the rooms is known. The transitions within the home are constrained by the topology.
The recognizer must be able to hypothesize the most likely sequence of rooms using features recorded by
the mobile phone.

## Data

The data is stored in files. See the [dataframe package](https://github.com/akualab/dataframe).

## Topology

The topology of the house is provided in the file `topology-24001.json`. Here are the first few lines:

```YAML
name: southcourt
arcs:
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


### The Corpus

The data is organized by sessions. Each session corresponds to a user (eg. 24001) that collects data using
an Android device in a house. We collected data from various sensors using a fixed sampling period of 4 seconds.
The measurements include:

* Accelerometer (x,y,z)
* Gyroscope (x,y,z)
* Magnetometer (x,y,z)
*  Received Signal Strength Indication (RSSI) from various wireless stationary base stations located around the home.

The length of the wifi RSSI vector depends on how many access points were available at eash site. The RSSI measurements ar in dbm units. Missing values due to weak signal strength are filled with -100 dbm. In some cases we used heuristics to interpolate missing values.

The original data was sampled with an approximate period of 4 seconds. We processed the data to have an exact period of 4 secs by repeating or dropping some of the samples. We refer to each sample as a *frame*.

The data is labeled with the name of the room in which the data was collected. The labels were entered manually
by a person in real-time using a simple user-interface on the Android device. Therefore the accuracy of the labels near the transitions between rooms is not precise. Subjects were asked to set the label at the time they were entering a room.

### Train One Gaussian per Room

We model the RSSI distribution using a multivariate Gaussian model associated with each room. We name each model with
corresponding room name.

Install gjoa: `go install github.com/akualab/gjoa/gjoa`

Type `gjoa` to see usage info.

Let's add a self transition to all the nodes.
```
gjoa -g=3 -c=graph0.yaml graph
```

Take a look at the new graph: `out/topology-24001-self.json`

Train hmm: `gjoa train` uses the default config file `gjoa.yaml`.

You should see the model trained from Gaussians in `out/hmm0.json`.

To decode run `gjoa -c test.yaml decode` which uses the config file `test.yaml`.

### Expand graph to model transitions.

In the previous experiment we use the same model inside the room and during room transitions. Clearly the features will be quite different when the person is moving from room to room compared to when he is static inside the room. To improve teh model, we insert a new node between each room transition. Once we expand teh graph we will need to create an initial alignment to map teh original labels to the new labels. For example, if we had a transition A->B, we will now have A->AB->B. To train the initial model we will use alignments that  assign the last X frames of A to AB. Once we have the models trained with alignments we will run forward-backward training to improve the models.

Let's use the subcommand `graph` to expand the graph by inserting a new node between the original nodes.

```
gjoa -g=3 -c=graph1.yaml graph
```

Now compare the original and expanded graphs:

Original:

```JSON
{
  "arcs": {
    "LIVING": {
      "KITCHEN": 3,
      "BACKYARD": 1
    },
    "KITCHEN": {
      "LIVING": 3,
      "DINING": 3
    },
...
```

Expanded:

```JSON
{
  "arcs": {
    "LIVING-KITCHEN": {
      "LIVING-KITCHEN": 0.5,
      "KITCHEN": 0.5
    },
    "LIVING-BACKYARD": {
      "LIVING-BACKYARD": 0.5,
      "BACKYARD": 0.5
    },
    "LIVING": {
      "LIVING-KITCHEN": 0.75,
      "LIVING-BACKYARD": 0.25
    },
    "KITCHEN-LIVING": {
      "LIVING": 0.5,
      "KITCHEN-LIVING": 0.5
    },
    "KITCHEN-DINING": {
      "KITCHEN-DINING": 0.5,
...
```

Now let's train again using the expanded graph.

```
gjoa -g=3 -c=train-expanded.yaml train
```

the new model file is out/hmm2.json which has 51 states.

## Search

```
gjoa -g=3 -c test-expanded.yaml decode
```

This will write a results file in `out/results-expanded.json` which includes a list of result objects:

```JSON
{"batchid":"test-24001-010","ref":["BED1","BED1","BED1","BED1","BED1","BED1","BED4","BED4","BED2","BED2","BED2","BED2","BED4","BED4","BED4","BED4","BED4","BED4","DINING","DINING","DINING","DINING","BACKYARD","BACKYARD","BACKYARD","BACKYARD","BACKYARD","DINING","DINING","DINING","DINING","DINING","DINING","DINING","DINING","DINING","DINING","DINING","DINING","DINING","DINING","DINING","DINING","DINING","DINING","DINING","DINING","KITCHEN","KITCHEN","KITCHEN","KITCHEN","KITCHEN","KITCHEN","KITCHEN","KITCHEN","KITCHEN","KITCHEN","KITCHEN","DINING","DINING","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5"],"hyp":["BED1","BED1","BED1","BED1-BATH1","BED1-BATH1","BATH1","BATH1-BED4","BED4","BED4-BED2","BED2","BED2","BED2","BED2-BED4","BED4","BED4-DINING","DINING","DINING-BED4","BED4","BED4-DINING","DINING","DINING","DINING-BACKYARD","BACKYARD","BACKYARD-DINING","BACKYARD-DINING","BACKYARD-DINING","BACKYARD-DINING","BACKYARD-DINING","DINING","DINING-BED4","BED4","BED4-DINING","DINING","DINING","DINING","DINING","DINING","DINING-BED4","BED4","BED4-DINING","DINING","DINING","DINING-BED4","DINING-BED4","BED4","BED4-DINING","DINING","DINING-KITCHEN","KITCHEN","KITCHEN-LIVING","KITCHEN-LIVING","KITCHEN-LIVING","KITCHEN-LIVING","KITCHEN-LIVING","LIVING","LIVING-KITCHEN","KITCHEN","KITCHEN-DINING","DINING","DINING-BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5","BED5"]}
```

The label `batchid` identifies the session, `ref` is an array of the original labels entered by a person (one label per frame), `hyp` is an array of the labels hypothesized by the algorithm (one per frame).

This raw file can now be processed to extract mertics at the session and data set levels. We use the `gjoa tally` command to compute the metrics.

**TIP:** If the results file is not specified, the results wil be written to stdout without formatting makign it easy to grep.


## Results
