#!/bin/bash

# https://github.com/sunniesuhyoung/DST


# Common arguments to all steps
im_size='256'
content_path='style/4.jpg'
style_path='example/f2.png'

# 1. Run NBB to get correspondences
results_dir='f22/NBBresults'
python NBB/main.py --results_dir ${results_dir} --imageSize ${im_size} --fast \
  --datarootA ${content_path} --datarootB ${style_path}


# 2. Clean (NBB) points
content_pts_path='f22/NBBresults/correspondence_A.txt'
style_pts_path='f22/NBBresults/correspondence_B.txt'
activation_path='f22/NBBresults/correspondence_activation.txt'
output_path='f22/CleanedPts'
NBB='1'
max_num_points='80'
b='10'

python cleanpoints.py ${content_path} ${style_path} ${content_pts_path} \
  ${style_pts_path} ${activation_path} ${output_path} \
  ${im_size} ${NBB} ${max_num_points} ${b}


# 3. Run DST
content_pts_path='f22/CleanedPts/correspondence_A.txt'
style_pts_path='f22/CleanedPts/correspondence_B.txt'
output_dir='f22/DSTresults'
output_prefix='f22'
max_iter='150'
checkpoint_iter='50'
content_weight='8' # alpha
warp_weight='0.5' # beta
reg_weight='50' # gamma
optim='sgd'
lr='0.3'
verbose='1'
save_intermediate='1'
save_extra='1'
device='cuda:0' # cpu or cuda

python -W ignore main.py ${content_path} ${style_path} ${content_pts_path} ${style_pts_path} \
  ${output_dir} ${output_prefix} ${im_size} ${max_iter} \
  ${checkpoint_iter} ${content_weight} ${warp_weight} ${reg_weight} ${optim} \
  ${lr} ${verbose} ${save_intermediate} ${save_extra} ${device}
