# apollo_data_analysis

## Installation

```
$ conda config --add channels conda-forge
$ conda create -n darts python=3.10 obspy cartopy pytest pytest-json
$ conda activate darts 
$ conda install pip
$ git clone https://github.com/isas-yamamoto/obspy.git
$ cd obspy
$ pip install .
$ conda install -n darts ipykernel --update-deps --force-reinstall
$ conda install pandas
```

## Usage

Read sample.ipynb