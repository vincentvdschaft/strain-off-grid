#!/bin/bash
make_cardiac_phantom
simulate_phantom --phantom polygon --n-tx 5 --angle-delta-deg 0.0 --scatterers 10000 --prf 2000 --focal-type diverging --n-frames 20 --out source/simulated_phantom_ischemic.hdf5

