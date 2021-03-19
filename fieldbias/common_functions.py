#import pyfftw as fftw
import os
import struct
from collections import namedtuple
#from nbodykit.source.mesh import ArrayMesh
import psutil
import numpy as np
import gc
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray

__all__ = ['readGadgetSnapshot', 'GadgetHeader']

__GadgetHeader_fmt = '6I6dddii6Iiiddddii6Ii'

GadgetHeader = namedtuple('GadgetHeader', \
        'npart mass time redshift flag_sfr flag_feedback npartTotal flag_cooling num_files BoxSize Omega0 OmegaLambda HubbleParam flag_age flag_metals NallHW flag_entr_ics')
def get_memory():
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss/1e9, "GB is current memory usage")  # in bytes 


def readGadgetSnapshot(filename, read_pos=False, read_vel=False, read_id=False,\
        read_mass=False, print_header=False, single_type=-1, lgadget=False):
    """
    This function reads the Gadget-2 snapshot file.

    Parameters
    ----------
    filename : str
        path to the input file
    read_pos : bool, optional
        Whether to read the positions or not. Default is false.
    read_vel : bool, optional
        Whether to read the velocities or not. Default is false.
    read_id : bool, optional
        Whether to read the particle IDs or not. Default is false.
    read_mass : bool, optional
        Whether to read the masses or not. Default is false.
    print_header : bool, optional
        Whether to print out the header or not. Default is false.
    single_type : int, optional
        Set to -1 (default) to read in all particle types.
        Set to 0--5 to read in only the corresponding particle type.
    lgadget : bool, optional
        Set to True if the particle file comes from l-gadget.
        Default is false.

    Returns
    -------
    ret : tuple
        A tuple of the requested data.
        The first item in the returned tuple is always the header.
        The header is in the GadgetHeader namedtuple format.
    """
    blocks_to_read = (read_pos, read_vel, read_id, read_mass)
    ret = []
    with open(filename, 'rb') as f:
        f.seek(4, 1)
        h = list(struct.unpack(__GadgetHeader_fmt, \
                f.read(struct.calcsize(__GadgetHeader_fmt))))
        if lgadget:
            h[30] = 0
            h[31] = h[18]
            h[18] = 0
            single_type = 1
        h = tuple(h)
        header = GadgetHeader._make((h[0:6],) + (h[6:12],) + h[12:16] \
                + (h[16:22],) + h[22:30] + (h[30:36],) + h[36:])
        if print_header:
            print(header)
        if not any(blocks_to_read):
            return header
        ret.append(header)
        f.seek(256 - struct.calcsize(__GadgetHeader_fmt), 1)
        f.seek(4, 1)
        #
        mass_npart = [0 if m else n for m, n in zip(header.mass, header.npart)]
        if single_type not in set(range(6)):
            single_type = -1
        #
        for i, b in enumerate(blocks_to_read):
            fmt = np.dtype(np.float32)
            fmt_64 = np.dtype(np.float64)
            item_per_part = 1
            npart = header.npart
            #
            if i < 2:
                item_per_part = 3
            elif i == 2:
                fmt = np.dtype(np.uint32)
                fmt_64 = np.dtype(np.uint64)
            elif i == 3:
                if sum(mass_npart) == 0:
                    ret.append(np.array([], fmt))
                    break
                npart = mass_npart
            #
            size_check = struct.unpack('I', f.read(4))[0]
            #
            block_item_size = item_per_part*sum(npart)
            if size_check != block_item_size*fmt.itemsize:
                fmt = fmt_64
            if size_check != block_item_size*fmt.itemsize:
                raise ValueError('Invalid block size in file!')
            size_per_part = item_per_part*fmt.itemsize
            #
            if not b:
                f.seek(sum(npart)*size_per_part, 1)
            else:
                if single_type > -1:
                    f.seek(sum(npart[:single_type])*size_per_part, 1)
                    npart_this = npart[single_type]
                else:
                    npart_this = sum(npart)
                data = np.fromstring(f.read(npart_this*size_per_part), fmt)
                if item_per_part > 1:
                    data.shape = (npart_this, item_per_part)
                ret.append(data)
                if not any(blocks_to_read[i+1:]):
                    break
                if single_type > -1:
                    f.seek(sum(npart[single_type+1:])*size_per_part, 1)
            f.seek(4, 1)
    #
    return tuple(ret)

def position_to_index(pos, lbox, nmesh):
    
    deltax = lbox/nmesh
    
    
    idvec = np.floor((pos)/deltax) 
    
    return (idvec%nmesh).astype('int16')

def diracdelta(i, j):
    if i == j:
        return 1
    else:
        return 0

# def delta_to_tidesq(delta_k, nmesh, lbox):
#     #Assumes delta_k is a pyfftw fourier-transformed density contrast field
#     #Computes the tidal tensor tau_ij = (k_i k_j/k^2  - delta_ij/3 )delta_k
#     #Returns it as an nbodykit mesh
#     kvals = np.fft.fftfreq(nmesh)*(2*np.pi*nmesh)/lbox
#     kvalsr = np.fft.rfftfreq(nmesh)*(2*np.pi*nmesh)/lbox
    
#     kx, ky, kz = np.meshgrid(kvals, kvals, kvalsr)
    
    
#     knorm = kx**2 + ky**2 + kz**2
#     knorm[0][0][0] = 1
#     klist = [[kx, kx], [kx, ky], [kx, kz], [ky, ky], [ky, kz], [kz, kz]]
    
#     del kx, ky, kz
#     gc.collect()
    
    
#     #Compute the symmetric tide at every Fourier mode which we'll reshape later
    
#     #Order is xx, xy, xz, yy, yz, zz
    
    
#     jvec = [[0,0], [0,1], [0,2], [1,1], [1,2], [2,2]]
#     tidesq = np.zeros(shape=(len(kvals), len(kvals), len(kvals)))

#     for i in range(len(klist)):
#         fft_tide = np.array((klist[i][0]*klist[i][1]/knorm - diracdelta(jvec[i][0], jvec[i][1])/3.) * (delta_k), dtype='complex64')
#         print(fft_tide.shape, fft_tide.dtype)
#         fft_tide = fftw.byte_align(fft_tide, fftw.simd_alignment, dtype='complex64')
#         real_out = fft_tide.view('float32')[:,:,:nmesh]
#         complex_in = fft_tide[:,:,:]
#         print(fft_tide.shape, real_out.shape, complex_in.shape)
#         transform = fftw.FFTW(complex_in, real_out, axes=(0,1,2),direction='FFTW_BACKWARD', threads=NTHREADS)
#         transform()
#         tidesq += real_out**2
#         if jvec[i][0] != jvec[i][1]:
#             tidesq+= real_out**2
            
#     del fft_tide
#     gc.collect()
     
#     return ArrayMesh(tidesq, BoxSize=lbox).to_real_field()


def delta_to_gradsqdelta(delta_k, nmesh, lbox):
    
    kvals = np.fft.fftfreq(nmesh)*(2*np.pi*nmesh)/lbox
    kvalsr = np.fft.rfftfreq(nmesh)*(2*np.pi*nmesh)/lbox
    
    kx, ky, kz = np.meshgrid(kvals, kvals, kvalsr)
    
    
    knorm = kx**2 + ky**2 + kz**2
    knorm[0][0][0] = 1
    
    ksqdelta = knorm*delta_k
    
    ksqdelta = fftw.byte_align(ksqdelta, dtype='complex64')
    
    gradsqdelta = fftw.interfaces.numpy_fft.irfftn(ksqdelta, axes=[0,1,2], threads=-1)

    
    return ArrayMesh(gradsqdelta, BoxSize=lbox).to_real_field()

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
    #Assumes delta_k is a pyfftw fourier-transformed density contrast field
    #Computes the tidal tensor tau_ij = (k_i k_j/k^2  - delta_ij/3 )delta_k
    #Returns it as an nbodykit mesh
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
        karray = (klist[i][0]*klist[i][1]/knorm - diracdelta(jvec[i][0], jvec[i][1])/3.)
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


