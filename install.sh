#!/bin/bash

git submodule update --init
apt-get -y install pipenv
pipenv install
pipenv run pip3 install -e transformers