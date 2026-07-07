#!/bin/bash

evaluate_strain out/sweep/ischemic_00/* --direction 0 --no-show --out out/ischemic && evaluate_strain out/sweep/ischemic_00/* --direction 1 --no-show && plot_ring_strain_curves