python3 data/prepare_train_data.py /projects/venous-malformations-hpc/data/msz/sfml/kitti \
 --dataset-format 'kitti_raw' --dump-root /projects/venous-malformations-hpc/data/msz/sfml/formatted_data \
 --width 416 --height 128 --num-threads 4 --with-depth --with-pose