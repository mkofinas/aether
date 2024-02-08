## Dynamic gravitational field

[![Gravitational Field Dataset](https://img.shields.io/badge/Zenodo-Gravitational%20Field%20Dataset-blue?logo=zenodo)](https://doi.org/10.5281/zenodo.10634923)

#### Download

You can download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.10634923).

#### Re-create the dataset

Alternatively, you can re-create the dataset by running the following commands:

```python
python ../electrostatic/dataset/generate_dataset.py \
  --simulation gravitational_field --dim 3 --static-balls 1 --n-balls 5 \
  --strength 1e1 --data_dir dataset/data/gravitational_field_3d --num-train 50000 \
  --num-valid 10000 --num-test 10000 --sample-freq 100 --length 5000 --length-test 5000
python dataset/convert_dynamic_gravitational_dataset.py
```
These commands generate the (dynamic) gravitational field and convert it to Pytorch format.
