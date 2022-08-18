#!/bin/bash

# Common arguments to all steps
im_size='256'
content_path='face/1.jpg'
style_path='imgs/man1.png'

# 1. Run NBB to get correspondences
results_dir='example/NBBresults'
python NBB/main.py --results_dir ${results_dir} --imageSize ${im_size} --fast \
  --datarootA ${content_path} --datarootB ${style_path}


# 2. Clean (NBB) points
content_pts_path='example/NBBresults/correspondence_A.txt'
style_pts_path='example/NBBresults/correspondence_B.txt'
activation_path='example/NBBresults/correspondence_activation.txt'
output_path='example/CleanedPts'
NBB='1'
max_num_points='256'
b='5'

python cleanpoints.py ${content_path} ${style_path} ${content_pts_path} \
  ${style_pts_path} ${activation_path} ${output_path} \
  ${im_size} ${NBB} ${max_num_points} ${b}


# 3. Run V_P-demo
python face_demo(RADCloss).py
