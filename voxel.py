import os
import platform

VOXEL_PYTHON_VERSION = "3.9.6"

__version__ = VOXEL_PYTHON_VERSION
moduledir = os.path.dirname(os.path.realpath(__file__))
# if platform.system() == "Windows":
#     libname = "cvoxelise.pyd"

import cvoxelise as lib


def Read_stl(filename):
    """
    voxel.write(filename)

    Read mesh data in the form of an <*.stl> file.
    
    Return a list value.
    """
    if not os.path.exists(filename):
        print(filename," does not exists !!!")
        return None
    [coordV,coordN,stlname] = lib.Read_stl(filename)
    return [coordV,coordN,stlname]

def stlGetFormat(filename):
    """
    voxel.stlGetFormat(filename)

    Get stl file format : ascii or binary.
    
    Return a string value.
    """
    stlformat = lib.stlGetFormat(filename)
    return stlformat


class Voxel:
    """
    Voxel class.  
    1. voxel.Voxel(x,y,z)
    2. voxel.Voxel() and then Voxel.initSize(x, y, z)
    """
    def __init__(self,*size):
        """
        Voxel class initializtion function
        """
        self.voxel = lib.Voxel(*size)
        self._Nx, self._Ny, self._Nz = size
    
    @property
    def Nx(self):
        return self._Nx
    @property
    def Ny(self):
        return self._Ny
    @property
    def Nz(self):
        return self._Nz

    @property
    def gridCOx(self):
        return self.voxel.gridCOx
    @property
    def gridCOy(self):
        return self.voxel.gridCOy
    @property
    def gridCOz(self):
        return self.voxel.gridCOz
    @property
    def voxelgrid(self):
        """
        Voxel.voxelgrid  ———— Grid array of all nodes
        """
        return self.voxel.voxelgrid
    @property
    def nnod(self):
        """
        Voxel.nnod  ———— Number of all nodes
        """
        return self.voxel.nnod
    @property
    def nele(self):
        """
        Voxel.ele  ———— Number of all cube elements
        """
        return self.voxel.nele
    @property
    def ele_nod(self):
        """
        Voxel.ele_nod  ———— Element array store node-tags (int32, begins with 0)
        ([e1n1,e1n2,...,e1n8],
         [e2n1,e2n2,...,e2n8],...) 
        """
        return self.voxel.ele_nod
    @property
    def nod_coor(self):
        """
        Voxel.nod_coor  ———— Relative coordinates of nodes (int32, begins with 0)
        ([n1x,n1y,n1z],
         [n2x,n2y,n2z],...)
        """
        return self.voxel.nod_coor
    @property
    def nod_coor_abs(self):
        """
        Voxel.nod_coor  ———— Absolute coordinates of nodes (int32, begins with bbox.min)
        ([n1x,n1y,n1z],
         [n2x,n2y,n2z],...)
        """
        return self.voxel.nod_coor_abs
    
    def initSize(self, x, y, z):
        """
        initialize grid size with int values "x" "y" "z"
        """
        self.voxel.initSize(x, y, z)
        self._Nx = self.voxel.NX
        self._Ny = self.voxel.NY
        self._Nz = self.voxel.NZ

    
    def VOXELISE(self, mesharg, ray="xyz", parallel=True):
        """
        Voxelise a 3D triangular-polygon mesh. parameter "mesharg" is a filename of <.stl> file
        or a numpy.ndarray stored all triangular-polygon mesh

        Return a numpy.ndarray(int) "voxelgrid" and active property bellow:

        "gridCOx",  "gridCOy",  "gridCOz"
        """
        grid = self.voxel.VOXELISE(mesharg,ray,parallel)
        return grid
    
    def gen_vox_info(self):
        """
        generate voxel mesh info for graph show. 
        call this function to active property bellow:

        "nnod",  "nele",  "ele_nod",  "nod_coor",  "nod_coor_abs"
        """
        self.voxel.gen_vox_info()
        return

    def save_mesh(self, filename):
        self.voxel.save_mesh(filename)
        return

if __name__=="__main__":
    
    datapath="../modeldata"
    datapath = os.path.abspath(datapath)+"/"
    filename = os.path.join(datapath,"test.stl")
    meshVertexs = Read_stl(filename)[0]

    import time
    mvoxel = Voxel(120,120,120)
    start=time.time()
    mvoxel.VOXELISE(meshVertexs,"xyz",True)
    print('C++ Voxelise time(P): {:6f}'.format(time.time()-start))

    mvoxel2 = Voxel(120,120,120)
    start=time.time()
    mvoxel2.VOXELISE(meshVertexs,"xyz",False)
    print('C++ Voxelise time(S): {:6f}'.format(time.time()-start))

    mvoxel.gen_vox_info()
    print(mvoxel.nele)
    print(mvoxel.nnod)
