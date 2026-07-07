#!/bin/bash
# Sync libraries excluding the .git and .venv directories

rsync -av --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude="*.hdf5" ~/2-files/pymodules/imagelib libs
rsync -av --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude="*.hdf5" ~/2-files/pymodules/plotlib libs
rsync -av --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude="*.hdf5" ~/2-files/pymodules/latextoolkit libs
rsync -av --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude="*.hdf5" ~/1-projects/channel_ulm/libs/zea libs
rsync -av --exclude='.git' --exclude='.venv' --exclude='__pycache__' --exclude="*.hdf5" ~/1-projects/storepari libs