CUDA_VISIBLE_DEVICES=0 \
python3 train.py /projects/venous-malformations-hpc/data/msz/sfml/formatted_data \
  -b4 -m0.2 -s0.1 --epoch-size 3000 --sequence-length 3 --log-output 
  --sequence-length 5 --with-gt --with-pose