# OpenLane-V2 data

Download the **Subset A images** and **Map Element Bucket** annotations following the [OpenLane-V2 v2.1 data instructions](https://github.com/OpenDriveLab/OpenLane-V2/tree/v2.1.0/data).

Make the dataset root available as `data/`:

```bash
ln -s /path/to/OpenLane-V2/data/OpenLane-V2 data
python tools/data_process.py
```

Expected layout:

```text
data/
├── data_dict_subset_A.json
├── train/
│   └── <scene>/
│       ├── info/<timestamp>-ls.json
│       └── ... camera images ...
├── val/
│   └── <scene>/...
├── data_dict_subset_A_train_lanesegnet.pkl
└── data_dict_subset_A_val_lanesegnet.pkl
```

The preparation script preserves native lane instance IDs and poses, and represents each centerline and boundary using 10 XYZ points. These IDs support the temporal supervision in the released training code. The configuration reads annotations and images from the same `data/` root.

The official validation split contains 4,806 frames in 150 scenes. Frame order and scene boundaries are used by the streaming sampler; keep them intact.
