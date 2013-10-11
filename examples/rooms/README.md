# Room Recogition using Statistical Models

The task is to recognize the trajectory of a mobile phone inside a house. The house is divided in "rooms"
and the topology of the rooms is known. The transitions within the home are constrained by the topology.
The recognizer must be able to hypothesize the most likely sequence of rooms using features recorded by
the mobile phone.

## Data

The data is stored in files using the following format:


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

To train the Gaussians we do:


## Search


## Results
