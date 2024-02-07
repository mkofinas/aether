
These scripts are adapted from [NRI](https://github.com/ethanfetaya/NRI).

Run the following to generate the dataset with the (static) electrostatic field and
convert it to Pytorch format:
```python
python generate_dataset.py --simulation electrostatic_field --dim 2 --static_field \
  --static-balls 20 --strength 1.0 --data_dir electrostatic_field --num-train 50000 \
  --num-valid 10000 --num-test 10000 --sample-freq 100 --length 5000 --length-test 5000
python convert_static_electrostatic_dataset.py
```

Run the following to generate the dataset with the (dynamic) gravitational field and
convert it to Pytorch format:
```python
python generate_dataset.py --simulation gravitational_field --dim 3 --static-balls 1 \
  --n-balls 5 --strength 1e1 --data_dir gravitational_field_3d --num-train 50000 \
  --num-valid 10000 --num-test 10000 --sample-freq 100 --length 5000 --length-test 5000
python convert_dynamic_gravitational_dataset.py
```
