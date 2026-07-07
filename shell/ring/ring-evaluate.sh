#!/bin/bash

evaluate_strain out/sweep/ring_00/* --direction 0 --no-show && evaluate_strain out/sweep/ring_00/* --direction 1 --no-show && plot_ring_strain_curves