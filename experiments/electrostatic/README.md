## Electrostatic field

[![3D Charged Particles Dataset](https://img.shields.io/badge/Zenodo-Electrostatic%20Field%20Dataset-blue?logo=zenodo)](https://doi.org/10.5281/zenodo.10631646)

#### Download

You can download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.10631646).

#### Re-create the dataset

Alternatively, you can re-create the dataset by running the following commands:

```python
python dataset/generate_dataset.py --simulation electrostatic_field --dim 2 \
  --static_field --static-balls 20 --strength 1.0 \
  --data_dir dataset/data/electrostatic_field --num-train 50000 --num-valid 10000 \
  --num-test 10000 --sample-freq 100 --length 5000 --length-test 5000
python dataset/convert_static_electrostatic_dataset.py
```
These commands generate the (static) electrostatic field dataset and convert it to
Pytorch format.

