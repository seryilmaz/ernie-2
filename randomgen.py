from __future__ import print_function
import numpy as np

np.set_printoptions(threshold = np.nan)

filename = 'sparsegen7.py'
f = open(filename,'w')
#generates 9 blocks- each 256x56
indices = np.zeros((256*56*9,2),dtype=np.int8)
vec= np.arange(0,2304)
vec2= np.arange(0,504)
perm = np.random.permutation(vec)
perm2 = np.random.permutation(vec2)
ss='import numpy as np\nindices_layer3=np.array(['
for k in range(0,9):
  print(k)
  for i in range(0,256):
    for j in range(0,56):
      if i*j*k<8*255*55:
        ss=ss+'['+str(perm[k*256+i])+','+str(perm2[k*56+j])+'],'
      else: 
        ss=ss+'['+str(perm[k*256+i])+','+str(perm2[k*56+j])+']'
ss=ss+'])'
f.write(ss)   
f.close()

f = open(filename,'a')
#generates 2 blocks- each 252x128
indices = np.zeros((252*128*2,2),dtype=np.int8)
vec= np.arange(0,252*2)
vec2= np.arange(0,256)
perm = np.random.permutation(vec)
perm2 = np.random.permutation(vec2)
ss='\nindices_layer4=np.array(['
for k in range(0,2):
  print(k)
  for i in range(0,252):
    for j in range(0,128):
      if i*j*k<1*251*127:
        ss=ss+'['+str(perm[k*252+i])+','+str(perm2[k*128+j])+'],'
      else:
        ss=ss+'['+str(perm[k*252+i])+','+str(perm2[k*128+j])+']])'
f.write(ss)   
f.close()
