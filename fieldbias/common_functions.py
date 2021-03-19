import os
import struct
from collections import namedtuple
import psutil
import numpy as np

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

def kroneckerdelta(i, j):
    if i == j:
        return 1
    else:
        return 0




