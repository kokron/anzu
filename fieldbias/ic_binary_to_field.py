import numpy as np
from glob import glob
import time
import sys
import os
import threading
import gc
import yaml

NTHREADS = threading.active_count()
print(NTHREADS)

'''
This script generates a numpy array of the noiseless ICs from all of the individual slabs produced during a run of 2LPTIC.

'''

    
# icdir = glob('/u/ki/jderose/desims/tinkers_emu/Box000/ics_z9/ics.*')
# icdir.sort()

# fdir = glob('/nfs/slac/g/ki/ki22/cosmo/beckermr/tinkers_emu/Box000/output/snapdir_009/snapshot_009.*')

yamldir = sys.argv[1]
configs = yaml.load(open(yamldir, 'r'))

icdir = configs['icdir']
outdir = configs['outdir']
nmesh = configs['nmesh_in']
# try: testvar = sys.argv[2]
# except: testvar = ''

# if 'Test' in testvar:
#     testbox = int(sys.argv[3])
#     icdir =  '/home/users/jderose/uscratch/tinkers_emu_ics/%sBox%03d-%03d/ics/'%(testvar, boxno, testbox)
#     outdir = '/home/users/kokron/scratch/ptbias_emu/%sBox%03d-%03d/'%(testvar,boxno,testbox)
# else:
#     icdir =  '/home/users/jderose/uscratch/tinkers_emu_ics/Box%03d/ics/'%(boxno)
#     outdir = '/home/users/kokron/scratch/ptbias_emu/Box%03d/'%(boxno)
print(icdir)
Nics = len(glob(icdir+'deltalin.*'))
print(Nics)
#make directory if not there
os.makedirs(outdir, exist_ok=True)
bigarr = []
start_time = time.time()

for i in range(Nics):
    test = np.fromfile(icdir+'deltalin.%s'%i)

    #bigboi[i*N:(i+1)*N] = test
    idx = np.where(test !=0)[0]
    bigarr.append(test[idx])
    if i%5 == 0:
        print(i, " took ", time.time() - start_time)
bigarr = np.array(bigarr)
bigflat = bigarr.flatten()
n=0
print("Loaded things in, reshaping")
for i in range(Nics):
    n+=len(bigflat[i])
try :
    n - nmesh**3 == 0
    bigmesh = np.zeros(n, dtype='float32')
except:
    print('Nmesh_in does not match the number of grid cells from files in the IC directory!')

c=0
for i in range(Nics):
    l = len(bigflat[i])
    bigmesh[c:c+l] = bigflat[i]
    c+=l
del bigflat, bigarr
gc.collect()
bigmesh = bigmesh.reshape(nmesh, nmesh, nmesh)
#Aemu
print('Made bigmesh')
sys.stdout.flush()
print('Linear component field is done, took ', time.time() - start_time)
sys.stdout.flush()
np.save(outdir+'linICfield', bigmesh)
