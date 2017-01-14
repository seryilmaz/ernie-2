import os
import numpy as np

print('\n\n...STARTED SWEEPING...\n\n')

for i in range(1):

  lr =0.03
  lr_decay= 0.1 
  momentum = 0.9 
  wd=0.0000
  stddev1=0.03
  stddev2=stddev1 
  print('\n\n\n\n')
  command = 'python train.py '+str(lr) + ' '+ str(lr_decay)+' '+ str(momentum)+ ' '+ str(wd) + ' '+ str(stddev1)+' '+str(stddev2)+ ' 300000 1 1 1'
  os.system(command)
  
  lr =0.03
  lr_decay= 0.1 
  momentum = 0.9 
  wd=0.0004
  stddev1=0.03
  stddev2=stddev1 #stddev1 #0.03 for no local
  print('\n\n\n\n')
  command = 'python train.py '+str(lr) + ' '+ str(lr_decay)+' '+ str(momentum)+ ' '+ str(wd) + ' '+ str(stddev1)+' '+str(stddev2)+ ' 300000 1 1 1'
  os.system(command)
  

