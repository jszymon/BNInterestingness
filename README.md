# Find patterns in data which are interesting with respect to background knowledge represented by a Bayesian network

## What you need:
* Python 3.x
* the numpy package

## How to run the code:
* unpack the source
* change to the top level source directory
* type 'python BNInterGUI.py'

## How to use:
* one window contains interesting patterns, the other, the Bayesian network
* example data and network is in the 'data' directory
  NOTE: the netowrk is automatically genrated and does not necessarily reflect true causality, it's better to build a network from scratch
* if no network is loaded initial independence is assumed
* two inference methods are available: sampling (good for large datasets) and exact (good for small datasets)
