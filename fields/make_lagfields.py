import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
import time
import gc
import sys
import h5py
import yaml
import os
from common_functions import get_memory, kroneckerdelta


def MPI_mean(array):
    '''
    Computes the mean of an array that is slab-decomposed across multiple processes.
    '''
    procsum = np.sum(array)*np.ones(1)
    recvbuf = None
    if rank==0:
        recvbuf = np.zeros(shape=[nranks,1])
    comm.Gather(procsum, recvbuf, root=0)
    if rank==0:
        fieldmean = np.ones(1)*np.sum(recvbuf)/nmesh**3
    else:
        fieldmean = np.ones(1)
    comm.Bcast(fieldmean, root=0)
    return fieldmean[0]

def delta_to_tidesq(delta_k, nmesh, lbox, rank, nranks, fft):
    '''
    Computes the square tidal field from the density FFT
    
    s^2 = s_ij s_ij

    where 

    s_ij = (k_i k_j / k^2 - delta_ij / 3 ) * delta_k

    Inputs:
    delta_k: fft'd density, slab-decomposed. 
    nmesh: size of the mesh
    lbox: size of the box
    rank: current MPI rank
    nranks: total number of MPI ranks
    fft: PFFT fourier transform object. Used to do the backwards FFT.

    Outputs: 
    tidesq: the s^2 field for the given slab.
    '''

    kvals = np.fft.fftfreq(nmesh)*(2*np.pi*nmesh)/lbox
    kvalsmpi = kvals[rank*nmesh//nranks:(rank+1)*nmesh//nranks]
    kvalsr = np.fft.rfftfreq(nmesh)*(2*np.pi*nmesh)/lbox

    kx, ky, kz = np.meshgrid(kvalsmpi,kvals,  kvalsr)
    if rank==0:
        print(kvals.shape, kvalsmpi.shape, kvalsr.shape, "shape of x, y, z")

    knorm = kx**2 + ky**2 + kz**2
    if knorm[0][0][0] == 0:
        knorm[0][0][0] = 1

    klist = [[kx, kx], [kx, ky], [kx, kz], [ky, ky], [ky, kz], [kz, kz]]

    del kx, ky, kz
    gc.collect()

    #Compute the symmetric tide at every Fourier mode which we'll reshape later

    #Order is xx, xy, xz, yy, yz, zz
    jvec = [[0,0], [0,1], [0,2], [1,1], [1,2], [2,2]]
    tidesq = np.zeros((nmesh//nranks,nmesh,nmesh), dtype='float32')

    if rank==0:
        get_memory()
    for i in range(len(klist)):
        karray = (klist[i][0]*klist[i][1]/knorm - kroneckerdelta(jvec[i][0], jvec[i][1])/3.)
        fft_tide = np.array(karray * (delta_k), dtype='complex64')

        #this is the local sij
        real_out = fft.backward(fft_tide)

        if rank==0:
            get_memory()
        # fft_tide = fftw.byte_align(fft_tide, fftw.simd_alignment, dtype='complex64')
        # real_out = fft_tide.view('float32')[:,:,:nmesh]
        # complex_in = fft_tide[:,:,:]
        # print(fft_tide.shape, real_out.shape, complex_in.shape)
        tidesq += 1.*real_out**2
        if jvec[i][0] != jvec[i][1]:
            tidesq+= 1.*real_out**2
            
        del fft_tide, real_out
        gc.collect()
    # pass
    return tidesq

def delta_to_gradsqdelta(delta_k, nmesh, lbox, rank, nranks, fft):
    '''
    Computes the density curvature from the density FFT
    
    nabla^2 delta = IFFT(-k^2 delta_k)

    Inputs:
    delta_k: fft'd density, slab-decomposed. 
    nmesh: size of the mesh
    lbox: size of the box
    rank: current MPI rank
    nranks: total number of MPI ranks
    fft: PFFT fourier transform object. Used to do the backwards FFT.

    Outputs: 
    real_gradsqdelta: the nabla^2delta field for the given slab.
    '''

    kvals = np.fft.fftfreq(nmesh)*(2*np.pi*nmesh)/lbox
    kvalsmpi = kvals[rank*nmesh//nranks:(rank+1)*nmesh//nranks]
    kvalsr = np.fft.rfftfreq(nmesh)*(2*np.pi*nmesh)/lbox

    kx, ky, kz = np.meshgrid(kvalsmpi,kvals,  kvalsr)
    if rank==0:
        print(kvals.shape, kvalsmpi.shape, kvalsr.shape, "shape of x, y, z")

    knorm = kx**2 + ky**2 + kz**2
    if knorm[0][0][0] == 0:
        knorm[0][0][0] = 1

    del kx, ky, kz
    gc.collect()
    
    #Compute -k^2 delta which is the gradient
    ksqdelta = -np.array(knorm * (delta_k), dtype='complex64')
    
    real_gradsqdelta = fft.backward(ksqdelta)

    
    return real_gradsqdelta
if __name__ == "__main__":
    yamldir = sys.argv[1]
    configs = yaml.load(open(yamldir, 'r'))

    lindir = configs['outdir']

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    nmesh = configs['nmesh_in'] 
    bigarr = []
    start_time = time.time()
    Lbox = configs['lbox']


    N = np.array([nmesh,nmesh,nmesh], dtype=int)

    fft= PFFT(MPI.COMM_WORLD, N, axes=(0,1,2), dtype='float32', grid=(-1,))

    try:
        bigmesh = np.load(lindir+'linICfield.npy', mmap_mode='r')
    except:
        print('Have you run ic_binary_to_field.py yet? Did not find the right file.')
    #print(rank*nmesh//nranks,(rank+1)*nmesh//nranks)
    u = newDistArray(fft, False)

    #Slab-decompose the noiseless ICs along the distributed array 
    u[:] = bigmesh[rank*nmesh//nranks:(rank+1)*nmesh//nranks, :, :].astype(u.dtype)

    #Compute the delta^2 field. This operation is local in real space.
    d2 = newDistArray(fft, False)
    d2[:] = u*u
    dmean = MPI_mean(d2)

    #Mean-subtract delta^2 
    d2 -= dmean
    if rank==0:
        print(dmean, ' mean deltasq')

    #Parallel-write delta^2 to hdf5 file
    d2.write(lindir+'mpi_icfields_nmesh%s.h5'%nmesh, 'deltasq', step=2)

    #Free up memory
    del d2,dmean
    gc.collect()

    #Write the linear density field to hdf5
    u.write(lindir+'mpi_icfields_nmesh%s.h5'%nmesh, 'delta', step=2)

    #Take a forward FFT of the linear density
    u_hat = fft.forward(u, normalize=True)
    if rank==0:
        print('Did backwards FFT')

    #Make a copy of FFT'd linear density. Will be used to make s^2 field.
    deltak = u_hat.copy()
    if rank==0:
        print('Did array copy')
    tinyfft = delta_to_tidesq(deltak, nmesh, Lbox, rank, nranks, fft)
    if rank==0:
        print('Made the tidesq field')

    #Populate output with distarray
    v = newDistArray(fft, False)

    v[:] = tinyfft

    #Need to compute mean value of tidesq to subtract:
    vmean = MPI_mean(v)
    if rank==0:
        print(vmean, ' mean tidesq')
    v -= vmean

    v.write(lindir+'mpi_icfields_nmesh%s.h5'%nmesh, 'tidesq', step=2)

    #clear up space yet again
    del v, tinyfft,vmean
    gc.collect()

    #Now make the nablasq field
    v = newDistArray(fft, False)
 
    nablasq = delta_to_gradsqdelta(deltak, nmesh, Lbox, rank, nranks, fft)

    v[:] = nablasq 

    v.write(lindir+'mpi_icfields_nmesh%s.h5'%nmesh, 'nablasq', step=2)
    #Moar space
    del u, bigmesh, deltak, u_hat,fft,v
    gc.collect()

    if configs['np_weightfields']:

        if rank==0:

            print('Wrote successfully! Now must convert to .npy files')
            print(time.time() - start_time," seconds!")
            get_memory() 
            f = h5py.File(lindir+'mpi_icfields_nmesh%s.h5'%nmesh, 'r')
            fkeys = list(f.keys())
            for key in fkeys:
                arr = f[key]['3D']['2']
                print('converting '+key+' to numpy array')
                np.save(lindir+'%s_np'%key, arr)
                print(time.time() - start_time, " seconds!")
                del arr
                gc.collect()
                get_memory()
            #Deletes the hdf5 file 
            os.system('rm '+lindir+'mpi_icfields_nmesh%s.h5'%nmesh)
    else: 
        if rank==0:
            print('Wrote successfully! Took %d seconds'%(time.time() - start_time))
# if rank==0:
# 	print(uj.shape)
# 	assert np.allclose(uj, u)
