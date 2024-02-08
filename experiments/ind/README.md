## Traffic scenes (inD)

You can request access for the dataset from https://levelxdata.com/ind-dataset/.
Note that the inD dataset is free for non-commercial use only.

In the following commands, we assume the dataset is downloaded at
`dataset/data/ind`. If you used a different directory, change the `data_dir`
accordingly.

```python
python dataset/generate_dataset.py --output_dir dataset/data/ind_processed/ --data_dir dataset/data/ind/data/

python dataset/generate_single_ind_dataset.py --data_dir dataset/data/ind_processed/ --output_dir dataset/data/ --original_data_dir dataset/data/ind/data/
ln -s single_ind_processed_2 dataset/data/single_ind_processed
cp dataset/data/ind/data/27_background.png dataset/data/single_ind_processed
cp dataset/data/ind_processed/train_data_stats dataset/data/single_ind_processed
cp dataset/data/ind_processed/train_speed_norm_stats dataset/data/single_ind_processed
```

### Single Map Dataset

Scenes [18-29]

From the processed dataset we use:
- Train: Last one
- Val: Entire
- Test: [0, 1, 2, 3]

### Class2Idx

Bicycle: 0
Car: 1
Pedestrian: 2
Truck: 1
