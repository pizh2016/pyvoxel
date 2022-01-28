import os
import numpy as np
import meshWriter



VOXEL_PYTHON_VERSION = "3.9.6"
PYPRINT = 1

__version__ = VOXEL_PYTHON_VERSION
moduledir = os.path.dirname(os.path.realpath(__file__))


def Read_stl(filename):
    """
    pvoxel.Read_stl(filename)

    Read mesh data in the form of an <*.stl> file.
    
    Return a list value.
    """
    if not os.path.exists(filename):
        print(filename," does not exists !!!")
        return None
    
    stlformat = stlGetFormat(filename)

    if stlformat=='ascii':
        [coordV,coordN,stlname] = __READ_stlascii(filename)
    elif stlformat=='binary':
        [coordV,coordN] = __READ_stlbinary(filename)
        stlname = 'unnamed_object'
    return [coordV,coordN,stlname]

def stlGetFormat(filename):
    """
    pvoxel.stlGetFormat(filename)

    Get stl file format : ascii or binary.
    
    Return a string value.
    """
    fid = open(filename,'rb')
    fid.seek(0,2)                # Go to the end of the file
    fidSIZE = fid.tell()         # Check the size of the file
    if (fidSIZE-84)%50 > 0:
        stlformat = 'ascii'
    else:
        fid.seek(0,0)            # go to the beginning of the file
        header  = fid.read(80).decode() 
        isSolid = header[0:5]=='solid'
        fid.seek(-80,2)          # go to the end of the file minus 80 characters
        tail       = fid.read(80)
        isEndSolid = tail.find(b'endsolid')+1

        if isSolid & isEndSolid:
            stlformat = 'ascii'
        else:
            stlformat = 'binary'
    fid.close()
    return stlformat


def __READ_stlascii(stlFILENAME):
    '''
    Read mesh data in the form of an ascii <*.stl> file
    '''
    fidIN = open(stlFILENAME,'r')
    fidCONTENTlist = [line.strip() for line in fidIN.readlines() if line.strip()]     #Read all the lines and Remove all blank lines
    fidCONTENT = np.array(fidCONTENTlist)
    fidIN.close()

    # Read the STL name
    line1 = fidCONTENT[0]
    if (len(line1) >= 7):
        stlname = line1[6:]
    else:
        stlname = 'unnamed_object'; 

    # Read the vector normals
    stringNORMALS = fidCONTENT[np.char.find(fidCONTENT,'facet normal')+1 > 0]
    coordN  = np.array(np.char.split(stringNORMALS).tolist())[:,2:].astype(float)

    # Read the vertex coordinates
    facetTOTAL       = stringNORMALS.size
    stringVERTICES   = fidCONTENT[np.char.find(fidCONTENT,'vertex')+1 > 0]
    coordVall = np.array(np.char.split(stringVERTICES).tolist())[:,1:].astype(float)
    cotemp           = coordVall.reshape((3,facetTOTAL,3),order='F')
    coordV    = cotemp.transpose(1,2,0)

    return [coordV,coordN,stlname]



def __READ_stlbinary(stlFILENAME):
    '''
    Read mesh data in the form of an binary <*.stl> file
    '''
    import struct
    # Open the binary STL file
    fidIN = open(stlFILENAME,'rb')
    # Read the header
    fidIN.seek(80,0)                                   # Move to the last 4 bytes of the header
    facetcount = struct.unpack('I',fidIN.read(4))[0]   # Read the number of facets (uint32:'I',4 bytes)

    # Initialise arrays into which the STL data will be loaded:
    coordN  = np.zeros((facetcount,3))
    coordV = np.zeros((facetcount,3,3))
    # Read the data for each facet:
    for loopF in np.arange(0,facetcount):
        tempIN = struct.unpack(12*'f',fidIN.read(4*12))# Read the data of each facet (float:'f',4 bytes)
        coordN[loopF,:]    = tempIN[0:3]   # x,y,z components of the facet's normal vector
        coordV[loopF,:,0] = tempIN[3:6]   # x,y,z coordinates of vertex 1
        coordV[loopF,:,1] = tempIN[6:9]   # x,y,z coordinates of vertex 2
        coordV[loopF,:,2] = tempIN[9:12]  # x,y,z coordinates of vertex 3 
        fidIN.read(2);   # Move to the start of the next facet.  Using file.read is much quicker than using seek 
    
    fidIN.close()
    return [coordV,coordN]


class Voxel:
    """
    Voxel class.  
    1. pvoxel.Voxel(x,y,z)
    2. pvoxel.Voxel() and then Voxel.initSize(x, y, z)
    """
    def __init__(self,*size):
        """
        Voxel class initializtion function
        """
        self._Nx, self._Ny, self._Nz = size
        self._nnod = 0
        self._nele = 0
        self.__meshXmin = 0
        self.__meshXmax = 0
        self.__meshYmin = 0
        self.__meshYmax = 0
        self.__meshZmin = 0
        self.__meshZmax = 0
    
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
        return self._gridCOx
    @property
    def gridCOy(self):
        return self._gridCOy
    @property
    def gridCOz(self):
        return self._gridCOz
    @property
    def voxelgrid(self):
        """
        Voxel.voxelgrid  ———— Grid array of all nodes
        """
        return self._voxelgrid
    @property
    def nnod(self):
        """
        Voxel.nnod  ———— Number of all nodes
        """
        return self._nnod
    @property
    def nele(self):
        """
        Voxel.ele  ———— Number of all cube elements
        """
        return self._nele
    @property
    def ele_nod(self):
        """
        Voxel.ele_nod  ———— Element array store node-tags (int32, begins with 0)
        ([e1n1,e1n2,...,e1n8],
         [e2n1,e2n2,...,e2n8],...) 
        """
        return self._ele_nod
    @property
    def nod_coor(self):
        """
        Voxel.nod_coor  ———— Relative coordinates of nodes (int32, begins with 0)
        ([n1x,n1y,n1z],
         [n2x,n2y,n2z],...)
        """
        return self._nod_coor
    @property
    def nod_coor_abs(self):
        """
        Voxel.nod_coor  ———— Absolute coordinates of nodes (int32, begins with bbox.min)
        ([n1x,n1y,n1z],
         [n2x,n2y,n2z],...)
        """
        return self._nod_coor_abs
    
    def initSize(self, x, y, z):
        """
        initialize grid size with int values "x" "y" "z"
        """
        self._Nx = x
        self._Ny = y
        self._Nz = z

    
    def VOXELISE(self, mesharg, ray="xyz", parallel=False):
        """
        Voxelise a 3D triangular-polygon mesh. parameter "mesharg" is a filename of <.stl> file
        or a numpy.ndarray stored all triangular-polygon mesh

        Return a numpy.ndarray(int) "voxelgrid" and active property bellow:

        "gridCOx",  "gridCOy",  "gridCOz"
        """
        ### READ INPUT PARAMETERS
        if isinstance(mesharg, str):
            meshXYZ = Read_stl(mesharg)[0]
        elif isinstance(mesharg, np.ndarray):
            meshXYZ = mesharg
        ### IDENTIFY THE MIN AND MAX X,Y,Z COORDINATES OF THE POLYGON MESH
        #======================================================
        self.__meshXmin = meshXYZ[:,0,:].min()
        self.__meshXmax = meshXYZ[:,0,:].max()
        self.__meshYmin = meshXYZ[:,1,:].min()
        self.__meshYmax = meshXYZ[:,1,:].max()
        self.__meshZmin = meshXYZ[:,2,:].min()
        self.__meshZmax = meshXYZ[:,2,:].max()
        ### The output grid will be defined by the coordinates in gridCOx, gridCOy, gridCOz
        voxwidth  = (self.__meshXmax-self.__meshXmin)/(self._Nx+1/2)
        self._gridCOx   = np.arange(self.__meshXmin+voxwidth/2, self.__meshXmax-voxwidth/2, voxwidth)  
        voxwidth  = (self.__meshYmax-self.__meshYmin)/(self._Ny+1/2)
        self._gridCOy   = np.arange(self.__meshYmin+voxwidth/2, self.__meshYmax-voxwidth/2, voxwidth)    
        voxwidth  = (self.__meshZmax-self.__meshZmin)/(self._Nz+1/2)
        self._gridCOz   = np.arange(self.__meshZmin+voxwidth/2, self.__meshZmax-voxwidth/2, voxwidth)  

        ### Check that the output grid is large enough to cover the mesh
        gridcheckX = 0
        gridcheckY = 0
        gridcheckZ = 0
        if (self._gridCOx.min()>self.__meshXmin or self._gridCOx.max()<self.__meshXmax):
            if self._gridCOx.min()>self.__meshXmin:
                self._gridCOx = np.insert(self._gridCOx,0,self.__meshXmin,0)
                gridcheckX = gridcheckX + 1
            if self._gridCOx.max()<self.__meshXmax:
                self._gridCOx = np.insert(self._gridCOx,self._gridCOx.size,self.__meshXmax,0)
                gridcheckX = gridcheckX + 2

        if (self._gridCOy.min()>self.__meshYmin or self._gridCOy.max()<self.__meshYmax):
            if self._gridCOy.min()>self.__meshYmin:
                self._gridCOy = np.insert(self._gridCOy,0,self.__meshYmin,0)
                gridcheckY = gridcheckY + 1
            if self._gridCOy.max()<self.__meshYmax:
                self._gridCOy = np.insert(self._gridCOy,self._gridCOy.size,self.__meshYmax,0)
                gridcheckY = gridcheckY + 2
        if (self._gridCOz.min()>self.__meshZmin or self._gridCOz.max()<self.__meshZmax):
            if self._gridCOz.min()>self.__meshZmin:
                self._gridCOz = np.insert(self._gridCOz,0,self.__meshZmin,0)
                gridcheckZ = gridcheckZ + 1
            if self._gridCOz.max()<self.__meshZmax:
                self._gridCOz = np.insert(self._gridCOz,self._gridCOz.size,self.__meshZmax,0)
                gridcheckZ = gridcheckZ + 2
        
        
        ### VOXELISE USING THE USER DEFINED RAY DIRECTION(S)
        #======================================================

        # Count the number of voxels in each direction:
        voxcountX = self._gridCOx.size
        voxcountY = self._gridCOy.size
        voxcountZ = self._gridCOz.size

        # Prepare logical array to hold the voxelised data:
        gridOUTPUT      = np.zeros( (voxcountX,voxcountY,voxcountZ,len(ray)), dtype=bool)
        countdirections = 0
        
        if parallel:
            if ray.find('x') + 1:
                gridOUTPUT[:,:,:,countdirections] = np.transpose(_VOXELISE_MP(self._gridCOy,self._gridCOz,self._gridCOx,meshXYZ[:,[1,2,0],:]),(2,0,1))
                countdirections = countdirections + 1
                if PYPRINT: print("gridOUTPUT x: ", gridOUTPUT.shape, gridOUTPUT.sum())
            if ray.find('y') + 1:
                gridOUTPUT[:,:,:,countdirections] = np.transpose(_VOXELISE_MP(self._gridCOz,self._gridCOx,self._gridCOy,meshXYZ[:,[2,0,1],:]),(1,2,0))
                countdirections = countdirections + 1
                if PYPRINT: print("gridOUTPUT y: ", gridOUTPUT.shape, gridOUTPUT.sum())
            if ray.find('z') + 1:
                gridOUTPUT[:,:,:,countdirections] = _VOXELISE_MP(self._gridCOx,self._gridCOy,self._gridCOz,meshXYZ)
                countdirections = countdirections + 1
                if PYPRINT: print("gridOUTPUT z: ", gridOUTPUT.shape, gridOUTPUT.sum())
        else:
            if ray.find('x') + 1:
                gridOUTPUT[:,:,:,countdirections] = np.transpose(_VOXELISE(self._gridCOy,self._gridCOz,self._gridCOx,meshXYZ[:,[1,2,0],:]),(2,0,1))
                countdirections = countdirections + 1
                if PYPRINT: print("gridOUTPUT x: ", gridOUTPUT.shape, gridOUTPUT.sum())
            if ray.find('y') + 1:
                gridOUTPUT[:,:,:,countdirections] = np.transpose(_VOXELISE(self._gridCOz,self._gridCOx,self._gridCOy,meshXYZ[:,[2,0,1],:]),(1,2,0))
                countdirections = countdirections + 1
                if PYPRINT: print("gridOUTPUT y: ", gridOUTPUT.shape, gridOUTPUT.sum())
            if ray.find('z') + 1:
                gridOUTPUT[:,:,:,countdirections] = _VOXELISE(self._gridCOx,self._gridCOy,self._gridCOz,meshXYZ)
                countdirections = countdirections + 1
                if PYPRINT: print("gridOUTPUT z: ", gridOUTPUT.shape, gridOUTPUT.sum())

        # Combine the results of each ray-tracing direction:
        if len(ray)>1:
            gridOUTPUT = np.sum(gridOUTPUT,axis=3)>=len(ray)/2

        ### RETURN THE OUTPUT GRID TO THE SIZE REQUIRED BY THE USER (IF IT WAS CHANGED EARLIER)
        #======================================================
        if gridcheckX == 1:
            gridOUTPUT = gridOUTPUT[1:,:,:]
            self._gridCOx    = self._gridCOx[1:]
        elif gridcheckX == 2:
            gridOUTPUT = gridOUTPUT[:-1,:,:]
            self._gridCOx    = self._gridCOx[:-1]
        elif gridcheckX == 3:
            gridOUTPUT = gridOUTPUT[1:-1,:,:]
            self._gridCOx    = self._gridCOx[1:-1]

        if gridcheckY == 1:
            gridOUTPUT = gridOUTPUT[:,1:,:]
            self._gridCOy    = self._gridCOy[1:]
        elif gridcheckY == 2:
            gridOUTPUT = gridOUTPUT[:,:-1,:]
            self._gridCOy    = self._gridCOy[:-1]
        elif gridcheckY == 3:
            gridOUTPUT = gridOUTPUT[:,1:-1,:]
            self._gridCOy    = self._gridCOy[1:-1]

        if gridcheckZ == 1:
            gridOUTPUT = gridOUTPUT[:,:,1:]
            self._gridCOz    = self._gridCOz[1:]
        elif gridcheckZ == 2:
            gridOUTPUT = gridOUTPUT[:,:,:-1]
            self._gridCOz    = self._gridCOz[:-1]
        elif gridcheckZ == 3:
            gridOUTPUT = gridOUTPUT[:,:,1:-1]
            self._gridCOz    = self._gridCOz[1:-1]
        
        if PYPRINT:
            print("voxelgrid sum: ", gridOUTPUT.shape, gridOUTPUT.sum())
        
        self._voxelgrid = gridOUTPUT
        return self._voxelgrid


    
    def gen_vox_info(self):
        """
        generate voxel mesh info for graph show. 
        call this function to active property bellow:

        "nnod",  "nele",  "ele_nod",  "nod_coor",  "nod_coor_abs"
        """
        if PYPRINT:
            print("gen_vox_info start")
        if not isinstance(self._voxelgrid, np.ndarray):
            print("voxelgrid is not generated, run Voxel.VOXELISE first !!!")
            return
        
        ele_vox = self._voxelgrid.transpose((2,1,0))
        self._nele = len(np.nonzero(ele_vox)[0])
        
        NX, NY, NZ = ele_vox.shape
        
        # compute the node vox matrix
        nod_vox = np.zeros((NX+1, NY+1, NZ+1))
        cube = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
        for c in cube:    
            nod_vox[c[0]:NX+c[0],:,:][:,c[1]:NY+c[1],:][:,:,c[2]:NZ+c[2]] = nod_vox[c[0]:NX+c[0],:,:][:,c[1]:NY+c[1],:][:,:,c[2]:NZ+c[2]] + ele_vox
            
        
        self._nnod = len(np.nonzero(nod_vox)[0])
        print("  nele:",self._nele,"nnod:",self._nnod)
        nod_vox = nod_vox > 0
        
        
        nod_order = np.zeros((NX+1, NY+1, NZ+1))
        # Generate the node matrix with global order
        nod_order[nod_vox] = range(self._nnod)
        
        
        
        cube_g = np.array([[1,1,1],[1,0,1],[1,0,0],[1,1,0],[0,1,1],[0,0,1],[0,0,0],[0,1,0]])
        ele_nod_global = np.zeros((NX*NY*NZ,8), dtype='int32')
        
        # Loop all elements and record the nodes order
        for i in range(8):
            ele_nod_i = nod_order[cube_g[i][0]:NX+cube_g[i][0],:,:][:,cube_g[i][1]:NY+cube_g[i][1],:][:,:,cube_g[i][2]:NZ+cube_g[i][2]]
            ele_nod_global[:,i] = ele_nod_i.flatten()
        self._ele_nod = ele_nod_global[(ele_vox>0).flatten(),:]
        
        
        # compute the coordinates of nodes
        nod_x, nod_y, nod_z = np.meshgrid(range(NY+1), range(NX+1), range(NZ+1))
        
        
        self._nod_coor = np.zeros((self._nnod, 3),dtype='int32')
        self._nod_coor[:,1] = nod_x[nod_vox]
        self._nod_coor[:,2] = nod_y[nod_vox]
        self._nod_coor[:,0] = nod_z[nod_vox]

        dx = (self.__meshXmax - self.__meshXmin) / (self._Nx + 0.5)
        dy = (self.__meshYmax - self.__meshYmin) / (self._Ny + 0.5)
        dz = (self.__meshZmax - self.__meshZmin) / (self._Nz + 0.5)

        lx = np.arange(self.__meshXmin, self.__meshXmax + dx, dx)
        ly = np.arange(self.__meshYmin, self.__meshYmax + dy, dy)
        lz = np.arange(self.__meshZmin, self.__meshZmax + dz, dz)
        
        self._nod_coor_abs = np.zeros((self._nnod, 3),dtype=float)
        self._nod_coor_abs[:, 0] = lx[self._nod_coor[:, 0]]
        self._nod_coor_abs[:, 1] = ly[self._nod_coor[:, 1]]
        self._nod_coor_abs[:, 2] = lz[self._nod_coor[:, 2]]
        
        if PYPRINT:
            print("gen_vox_info finished")
        
        return


    def save_mesh(self, filename):
        """
        save voxel mesh into <.inp> file. 
        
        Absolute coordinates of nodes will be write
        """
        exp = filename.split(".")[-1]
        if (exp == "inp"):
            meshWriter.write_inp(filename, 12, self._nod_coor_abs, self._ele_nod)
        elif (exp == "vtk"):
            meshWriter.write_vtk(filename, 12, self._nod_coor_abs, self._ele_nod)
        elif (exp == "vtu"):
            meshWriter.write_vtu(filename, 12, self._nod_coor_abs, self._ele_nod)

        return



def _VOXELISE(gridCOx,gridCOy,gridCOz,meshXYZ):
    # Count the number of voxels in each direction:
    voxcountX = gridCOx.size
    voxcountY = gridCOy.size
    voxcountZ = gridCOz.size

    # Prepare logical array to hold the voxelised data:
    gridOUTPUT = np.zeros( (voxcountX,voxcountY,voxcountZ), dtype=int)

    # Identify the min and max x,y,z coordinates (cm) of the mesh:
    meshXmin = meshXYZ[:,0,:].min()
    meshXmax = meshXYZ[:,0,:].max()
    meshYmin = meshXYZ[:,1,:].min()
    meshYmax = meshXYZ[:,1,:].max()
    meshZmin = meshXYZ[:,2,:].min()
    meshZmax = meshXYZ[:,2,:].max()

    # Identify the min and max x,y coordinates (pixels) of the mesh:
    meshXminp = np.where( np.abs(gridCOx-meshXmin)==np.abs(gridCOx-meshXmin).min() )[0][0]
    meshXmaxp = np.where( np.abs(gridCOx-meshXmax)==np.abs(gridCOx-meshXmax).min() )[0][0]
    meshYminp = np.where( np.abs(gridCOy-meshYmin)==np.abs(gridCOy-meshYmin).min() )[0][0]
    meshYmaxp = np.where( np.abs(gridCOy-meshYmax)==np.abs(gridCOy-meshYmax).min() )[0][0]

    # Make sure min < max for the mesh coordinates:
    if meshXminp > meshXmaxp:
        meshXminp,meshXmaxp = meshXmaxp,meshXminp
    if meshYminp > meshYmaxp:
        meshYminp,meshYmaxp = meshYmaxp,meshYminp

    # Identify the min and max x,y,z coordinates of each facet:
    meshXYZmin = np.min(meshXYZ,axis=2)
    meshXYZmax = np.max(meshXYZ,axis=2)

    #======================================================
    # VOXELISE THE MESH
    #======================================================
    correctionLIST = np.zeros((0,2),dtype=int) # N x 2
    # Loop through each x,y pixel.
    # The mesh will be voxelised by passing rays in the z-direction through
    # each x,y pixel, and finding the locations where the rays cross the mesh.
    for loopY in range(meshYminp,meshYmaxp+1):
        # - 1a - Find which mesh facets could possibly be crossed by the ray:
        possibleCROSSLISTy = np.where( (meshXYZmin[:,1]<=gridCOy[loopY]) & (meshXYZmax[:,1]>=gridCOy[loopY]) )[0]

        for loopX in range(meshXminp,meshXmaxp+1):       
            # - 1b - Find which mesh facets could possibly be crossed by the ray:
            possibleCROSSLIST = possibleCROSSLISTy[ (meshXYZmin[possibleCROSSLISTy,0]<=gridCOx[loopX]) & (meshXYZmax[possibleCROSSLISTy,0]>=gridCOx[loopX]) ]

            #Only continue the analysis if some nearby facets were actually identified
            if possibleCROSSLIST.size > 0:
                # - 2 - For each facet, check if the ray really does cross the facet rather than just passing it close-by:
                facetCROSSLIST = np.zeros(0,dtype=int)
                #----------
                # - 2 - Check for crossed facets:
                #----------
                #Only continue the analysis if some nearby facets were actually identified
                if possibleCROSSLIST.size > 0:  
                    # 判断给出的坐标是否在闭合的三角形中，向量叉积法
                    for loopCHECKFACET in possibleCROSSLIST.flatten():
                        #Check if ray crosses the facet.  This method is much (>>10 times) faster than using the built-in function 'inpolygon'.
                        #Taking each edge of the facet in turn, check if the ray is on the same side as the opposing vertex.
                        Y1predicted = meshXYZ[loopCHECKFACET,1,1] - ((meshXYZ[loopCHECKFACET,1,1]-meshXYZ[loopCHECKFACET,1,2]) * (meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,0])/(meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,2]))
                        YRpredicted = meshXYZ[loopCHECKFACET,1,1] - ((meshXYZ[loopCHECKFACET,1,1]-meshXYZ[loopCHECKFACET,1,2]) * (meshXYZ[loopCHECKFACET,0,1]-gridCOx[loopX])/(meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,2]))
                        
                        if (Y1predicted > meshXYZ[loopCHECKFACET,1,0] and YRpredicted > gridCOy[loopY]) or (Y1predicted < meshXYZ[loopCHECKFACET,1,0] and YRpredicted < gridCOy[loopY]):
                            #The ray is on the same side of the 2-3 edge as the 1st vertex.
                            Y2predicted = meshXYZ[loopCHECKFACET,1,2] - ((meshXYZ[loopCHECKFACET,1,2]-meshXYZ[loopCHECKFACET,1,0]) * (meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,1])/(meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,0]))
                            YRpredicted = meshXYZ[loopCHECKFACET,1,2] - ((meshXYZ[loopCHECKFACET,1,2]-meshXYZ[loopCHECKFACET,1,0]) * (meshXYZ[loopCHECKFACET,0,2]-gridCOx[loopX])/(meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,0]))
                        
                            if (Y2predicted > meshXYZ[loopCHECKFACET,1,1] and YRpredicted > gridCOy[loopY]) or (Y2predicted < meshXYZ[loopCHECKFACET,1,1] and YRpredicted < gridCOy[loopY]):
                                #The ray is on the same side of the 3-1 edge as the 2nd vertex.
                                Y3predicted = meshXYZ[loopCHECKFACET,1,0] - ((meshXYZ[loopCHECKFACET,1,0]-meshXYZ[loopCHECKFACET,1,1]) * (meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,2])/(meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,1]))
                                YRpredicted = meshXYZ[loopCHECKFACET,1,0] - ((meshXYZ[loopCHECKFACET,1,0]-meshXYZ[loopCHECKFACET,1,1]) * (meshXYZ[loopCHECKFACET,0,0]-gridCOx[loopX])/(meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,1]))
                                
                                if (Y3predicted > meshXYZ[loopCHECKFACET,1,2] and YRpredicted > gridCOy[loopY]) or (Y3predicted < meshXYZ[loopCHECKFACET,1,2] and YRpredicted < gridCOy[loopY]):
                                    #The ray is on the same side of the 1-2 edge as the 3rd vertex.
                                    #The ray passes through the facet since it is on the correct side of all 3 edges
                                    facetCROSSLIST = np.insert(facetCROSSLIST,facetCROSSLIST.size,loopCHECKFACET,0)
                
                    #----------
                    # - 3 - Find the z coordinate of the locations where the ray crosses each facet or vertex:
                    #----------
                    gridCOzCROSS = np.zeros(facetCROSSLIST.shape)
                    for loopFINDZ in facetCROSSLIST:
                        # METHOD:
                        # 1. Define the equation describing the plane of the facet.  For a more detailed outline of the maths, see:
                        # http://local.wasp.uwa.edu.au/~pbourke/geometry/planeeq/
                        #    Ax + By + Cz + D = 0
                        #    where  A = y0 (z1 - z2) + y1 (z2 - z0) + y2 (z0 - z1)
                        #           B = z0 (x1 - x2) + z1 (x2 - x0) + z2 (x0 - x1)
                        #           C = x0 (y1 - y2) + x1 (y2 - y0) + x2 (y0 - y1)
                        #           D = - x0 (y1 z2 - y2 z1) - x1 (y2 z0 - y0 z2) - x2 (y0 z1 - y1 z0)
                        # 2. For the x and y coordinates of the ray, solve these equations to find the z coordinate in this plane.

                        planecoA = meshXYZ[loopFINDZ,1,0]*(meshXYZ[loopFINDZ,2,1]-meshXYZ[loopFINDZ,2,2]) + meshXYZ[loopFINDZ,1,1]*(meshXYZ[loopFINDZ,2,2]-meshXYZ[loopFINDZ,2,0]) + meshXYZ[loopFINDZ,1,2]*(meshXYZ[loopFINDZ,2,0]-meshXYZ[loopFINDZ,2,1])
                        planecoB = meshXYZ[loopFINDZ,2,0]*(meshXYZ[loopFINDZ,0,1]-meshXYZ[loopFINDZ,0,2]) + meshXYZ[loopFINDZ,2,1]*(meshXYZ[loopFINDZ,0,2]-meshXYZ[loopFINDZ,0,0]) + meshXYZ[loopFINDZ,2,2]*(meshXYZ[loopFINDZ,0,0]-meshXYZ[loopFINDZ,0,1]) 
                        planecoC = meshXYZ[loopFINDZ,0,0]*(meshXYZ[loopFINDZ,1,1]-meshXYZ[loopFINDZ,1,2]) + meshXYZ[loopFINDZ,0,1]*(meshXYZ[loopFINDZ,1,2]-meshXYZ[loopFINDZ,1,0]) + meshXYZ[loopFINDZ,0,2]*(meshXYZ[loopFINDZ,1,0]-meshXYZ[loopFINDZ,1,1])
                        planecoD = - meshXYZ[loopFINDZ,0,0]*(meshXYZ[loopFINDZ,1,1]*meshXYZ[loopFINDZ,2,2]-meshXYZ[loopFINDZ,1,2]*meshXYZ[loopFINDZ,2,1]) - meshXYZ[loopFINDZ,0,1]*(meshXYZ[loopFINDZ,1,2]*meshXYZ[loopFINDZ,2,0]-meshXYZ[loopFINDZ,1,0]*meshXYZ[loopFINDZ,2,2]) - meshXYZ[loopFINDZ,0,2]*(meshXYZ[loopFINDZ,1,0]*meshXYZ[loopFINDZ,2,1]-meshXYZ[loopFINDZ,1,1]*meshXYZ[loopFINDZ,2,0])
                        
                        if abs(planecoC) < 1e-14: 
                            planecoC=0
                        else:
                            gridCOzCROSS[facetCROSSLIST==loopFINDZ] = (- planecoD - planecoA*gridCOx[loopX] - planecoB*gridCOy[loopY]) / planecoC

                    #Remove values of gridCOzCROSS which are outside of the mesh limits (including a 1e-12 margin for error).
                    gridCOzCROSS = gridCOzCROSS[ (gridCOzCROSS>=meshZmin-1e-12) & (gridCOzCROSS<=meshZmax+1e-12) ]
                    #Round gridCOzCROSS to remove any rounding errors, and take only the unique values:
                    gridCOzCROSS = np.round(gridCOzCROSS*1e12)/1e12
                    gridCOzCROSS = np.unique(gridCOzCROSS)
                    # print(gridCOzCROSS)

                    #----------
                    # - 4 - Label as being inside the mesh all the voxels that the ray passes through after crossing one facet before crossing another facet:
                    #----------
                    # Only rays which cross an even number of facets are voxelised
                    if gridCOzCROSS.size % 2 == 0:  
                        for loopASSIGN in np.arange( 1, (gridCOzCROSS.size/2)+1,dtype=int ):
                            voxelsINSIDE = ((gridCOz>gridCOzCROSS[2*loopASSIGN-2]) & (gridCOz<gridCOzCROSS[2*loopASSIGN-1]))
                            gridOUTPUT[loopX,loopY,voxelsINSIDE] = 1

                    elif gridCOzCROSS.size > 0:
                        # Remaining rays which meet the mesh in some way are not voxelised, but are labelled for correction later.
                        correctionLIST = np.insert( correctionLIST, correctionLIST.shape[0], [[loopX,loopY]], axis=0 )

    #======================================================
    # USE INTERPOLATION TO FILL IN THE RAYS WHICH COULD NOT BE VOXELISED
    #======================================================
    #For rays where the voxelisation did not give a clear result, the ray is computed by interpolating from the surrounding rays.
    countCORRECTIONLIST = correctionLIST.shape[0]

    if countCORRECTIONLIST>0:
        #If necessary, add a one-pixel border around the x and y edges of thearray.  
        #This prevents an error if the code tries to interpolate a ray at the edge of the x,y grid.
        if correctionLIST[:,0].min()==1 or correctionLIST[:,0].max()== gridCOx.size or correctionLIST[:,1].min()==1 or correctionLIST[:,1].max()==gridCOy.size:
            gridOUTPUT     = np.pad(gridOUTPUT,((1,1),(1,1),(0,0)),mode='constant')
            correctionLIST = correctionLIST + 1
        
        for loopC in np.arange( 0,countCORRECTIONLIST):
            voxelsforcorrection = np.sum( np.array([gridOUTPUT[correctionLIST[loopC,0]-1,correctionLIST[loopC,1]-1,:] ,\
                                        gridOUTPUT[correctionLIST[loopC,0]-1,correctionLIST[loopC,1],:]   ,\
                                        gridOUTPUT[correctionLIST[loopC,0]-1,correctionLIST[loopC,1]+1,:] ,\
                                        gridOUTPUT[correctionLIST[loopC,0],correctionLIST[loopC,1]-1,:]   ,\
                                        gridOUTPUT[correctionLIST[loopC,0],correctionLIST[loopC,1]+1,:]   ,\
                                        gridOUTPUT[correctionLIST[loopC,0]+1,correctionLIST[loopC,1]-1,:] ,\
                                        gridOUTPUT[correctionLIST[loopC,0]+1,correctionLIST[loopC,1],:]   ,\
                                        gridOUTPUT[correctionLIST[loopC,0]+1,correctionLIST[loopC,1]+1,:] ,\
                                        ]),axis=0 ) 
            voxelsforcorrection = (voxelsforcorrection>=4)
            gridOUTPUT[correctionLIST[loopC,0],correctionLIST[loopC,1],voxelsforcorrection] = 1
        #Remove the one-pixel border surrounding the array, if this was added
        #previously.
        if gridOUTPUT.shape[0]>gridCOx.size or gridOUTPUT.shape[1]>gridCOy.size:
            gridOUTPUT = gridOUTPUT[1:-1,1:-1,:]
    
    return gridOUTPUT


import multiprocessing
NUM_CORES = multiprocessing.cpu_count()
from joblib import Parallel, delayed

def _VOXELISE_MP(gridCOx,gridCOy,gridCOz,meshXYZ):
    # Count the number of voxels in each direction:
    voxcountX = gridCOx.size
    voxcountY = gridCOy.size
    voxcountZ = gridCOz.size

    # Prepare logical array to hold the voxelised data:
    gridshape = (voxcountX,voxcountY,voxcountZ)

    # Identify the min and max x,y,z coordinates (cm) of the mesh:
    meshXmin = meshXYZ[:,0,:].min()
    meshXmax = meshXYZ[:,0,:].max()
    meshYmin = meshXYZ[:,1,:].min()
    meshYmax = meshXYZ[:,1,:].max()
    

    # Identify the min and max x,y coordinates (pixels) of the mesh:
    meshXminp = np.where( np.abs(gridCOx-meshXmin)==np.abs(gridCOx-meshXmin).min() )[0][0]
    meshXmaxp = np.where( np.abs(gridCOx-meshXmax)==np.abs(gridCOx-meshXmax).min() )[0][0]
    meshYminp = np.where( np.abs(gridCOy-meshYmin)==np.abs(gridCOy-meshYmin).min() )[0][0]
    meshYmaxp = np.where( np.abs(gridCOy-meshYmax)==np.abs(gridCOy-meshYmax).min() )[0][0]

    # Make sure min < max for the mesh coordinates:
    if meshXminp > meshXmaxp:
        meshXminp,meshXmaxp = meshXmaxp,meshXminp
    if meshYminp > meshYmaxp:
        meshYminp,meshYmaxp = meshYmaxp,meshYminp

    #======================================================
    # VOXELISE THE MESH
    #======================================================

    parallel_obj = Parallel(n_jobs=int(NUM_CORES*0.5),verbose=0,backend='loky')
    result = parallel_obj(delayed(_ray_1direction)(i,gridshape,meshXYZ,gridCOx,gridCOy,gridCOz,meshXminp,meshXmaxp) for i in range(meshYminp,meshYmaxp+1))
    gridOUTPUT = np.array(result).sum(axis=0)
    
    return gridOUTPUT


def _ray_1direction(loopY,gridshape,meshXYZ,gridCOx,gridCOy,gridCOz,meshXminp,meshXmaxp):
    # - 1a - Find which mesh facets could possibly be crossed by the ray:
    # Identify the min and max x,y,z coordinates of each facet:
    meshXYZmin = np.min(meshXYZ,axis=2)
    meshXYZmax = np.max(meshXYZ,axis=2)
    meshZmin = meshXYZ[:,2,:].min()
    meshZmax = meshXYZ[:,2,:].max()
    gridTemp = np.zeros(gridshape, dtype=int)
    coory = gridCOy[loopY]
    possibleCROSSLISTy = np.where( (meshXYZmin[:,1]<=coory) & (meshXYZmax[:,1]>=coory) )[0]
    correctionLIST = np.zeros((0,2),dtype=int) # N x 2
    for loopX in range(meshXminp,meshXmaxp+1):       
        # - 1b - Find which mesh facets could possibly be crossed by the ray:
        coorx = gridCOx[loopX]
        possibleCROSSLIST = possibleCROSSLISTy[ (meshXYZmin[possibleCROSSLISTy,0]<=coorx) & (meshXYZmax[possibleCROSSLISTy,0]>=coorx) ]
        if possibleCROSSLIST.size > 0:
            # - 2 - For each facet, check if the ray really does cross the facet rather than just passing it close-by:
            facetCROSSLIST = np.zeros(0,dtype=int)
            if possibleCROSSLIST.size > 0:
                for loopCHECKFACET in possibleCROSSLIST.flatten():
                    #Check if ray crosses the facet.  This method is much (>>10 times) faster than using the built-in function 'inpolygon'.
                    #Taking each edge of the facet in turn, check if the ray is on the same side as the opposing vertex.
                    Y1predicted = meshXYZ[loopCHECKFACET,1,1] - ((meshXYZ[loopCHECKFACET,1,1]-meshXYZ[loopCHECKFACET,1,2]) * (meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,0])/(meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,2]))
                    YRpredicted = meshXYZ[loopCHECKFACET,1,1] - ((meshXYZ[loopCHECKFACET,1,1]-meshXYZ[loopCHECKFACET,1,2]) * (meshXYZ[loopCHECKFACET,0,1]-coorx)/(meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,2]))
                    
                    if (Y1predicted > meshXYZ[loopCHECKFACET,1,0] and YRpredicted > coory) or (Y1predicted < meshXYZ[loopCHECKFACET,1,0] and YRpredicted < coory):
                        #The ray is on the same side of the 2-3 edge as the 1st vertex.
                        Y2predicted = meshXYZ[loopCHECKFACET,1,2] - ((meshXYZ[loopCHECKFACET,1,2]-meshXYZ[loopCHECKFACET,1,0]) * (meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,1])/(meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,0]))
                        YRpredicted = meshXYZ[loopCHECKFACET,1,2] - ((meshXYZ[loopCHECKFACET,1,2]-meshXYZ[loopCHECKFACET,1,0]) * (meshXYZ[loopCHECKFACET,0,2]-coorx)/(meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,0]))
                        
                        if (Y2predicted > meshXYZ[loopCHECKFACET,1,1] and YRpredicted > coory) or (Y2predicted < meshXYZ[loopCHECKFACET,1,1] and YRpredicted < coory):
                            #The ray is on the same side of the 3-1 edge as the 2nd vertex.
                            Y3predicted = meshXYZ[loopCHECKFACET,1,0] - ((meshXYZ[loopCHECKFACET,1,0]-meshXYZ[loopCHECKFACET,1,1]) * (meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,2])/(meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,1]))
                            YRpredicted = meshXYZ[loopCHECKFACET,1,0] - ((meshXYZ[loopCHECKFACET,1,0]-meshXYZ[loopCHECKFACET,1,1]) * (meshXYZ[loopCHECKFACET,0,0]-coorx)/(meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,1]))
                            
                            if (Y3predicted > meshXYZ[loopCHECKFACET,1,2] and YRpredicted > coory) or (Y3predicted < meshXYZ[loopCHECKFACET,1,2] and YRpredicted < coory):
                                #The ray is on the same side of the 1-2 edge as the 3rd vertex.
                                #The ray passes through the facet since it is on the correct side of all 3 edges
                                facetCROSSLIST = np.insert(facetCROSSLIST,facetCROSSLIST.size,loopCHECKFACET,0)
                
                #----------
                # - 3 - Find the z coordinate of the locations where the ray crosses each facet or vertex:
                #----------
                gridCOzCROSS = np.zeros(facetCROSSLIST.shape)
                for loopFINDZ in facetCROSSLIST:
                    planecoA = meshXYZ[loopFINDZ,1,0]*(meshXYZ[loopFINDZ,2,1]-meshXYZ[loopFINDZ,2,2]) + meshXYZ[loopFINDZ,1,1]*(meshXYZ[loopFINDZ,2,2]-meshXYZ[loopFINDZ,2,0]) + meshXYZ[loopFINDZ,1,2]*(meshXYZ[loopFINDZ,2,0]-meshXYZ[loopFINDZ,2,1])
                    planecoB = meshXYZ[loopFINDZ,2,0]*(meshXYZ[loopFINDZ,0,1]-meshXYZ[loopFINDZ,0,2]) + meshXYZ[loopFINDZ,2,1]*(meshXYZ[loopFINDZ,0,2]-meshXYZ[loopFINDZ,0,0]) + meshXYZ[loopFINDZ,2,2]*(meshXYZ[loopFINDZ,0,0]-meshXYZ[loopFINDZ,0,1]) 
                    planecoC = meshXYZ[loopFINDZ,0,0]*(meshXYZ[loopFINDZ,1,1]-meshXYZ[loopFINDZ,1,2]) + meshXYZ[loopFINDZ,0,1]*(meshXYZ[loopFINDZ,1,2]-meshXYZ[loopFINDZ,1,0]) + meshXYZ[loopFINDZ,0,2]*(meshXYZ[loopFINDZ,1,0]-meshXYZ[loopFINDZ,1,1])
                    planecoD = - meshXYZ[loopFINDZ,0,0]*(meshXYZ[loopFINDZ,1,1]*meshXYZ[loopFINDZ,2,2]-meshXYZ[loopFINDZ,1,2]*meshXYZ[loopFINDZ,2,1]) - meshXYZ[loopFINDZ,0,1]*(meshXYZ[loopFINDZ,1,2]*meshXYZ[loopFINDZ,2,0]-meshXYZ[loopFINDZ,1,0]*meshXYZ[loopFINDZ,2,2]) - meshXYZ[loopFINDZ,0,2]*(meshXYZ[loopFINDZ,1,0]*meshXYZ[loopFINDZ,2,1]-meshXYZ[loopFINDZ,1,1]*meshXYZ[loopFINDZ,2,0])

                    if abs(planecoC) < 1e-14: 
                        planecoC=0
                    else:
                        gridCOzCROSS[facetCROSSLIST==loopFINDZ] = (- planecoD - planecoA*coorx - planecoB*coory) / planecoC

                #Remove values of gridCOzCROSS which are outside of the mesh limits (including a 1e-12 margin for error).
                gridCOzCROSS = gridCOzCROSS[ (gridCOzCROSS>=meshZmin-1e-12) & (gridCOzCROSS<=meshZmax+1e-12) ]
                #Round gridCOzCROSS to remove any rounding errors, and take only the unique values:
                gridCOzCROSS = np.round(gridCOzCROSS*1e12)/1e12
                gridCOzCROSS = np.unique(gridCOzCROSS)

                if gridCOzCROSS.size % 2 == 0:  
                    for loopASSIGN in np.arange( 1, (gridCOzCROSS.size/2)+1,dtype=int ):
                        voxelsINSIDE = ((gridCOz>gridCOzCROSS[2*loopASSIGN-2]) & (gridCOz<gridCOzCROSS[2*loopASSIGN-1]))
                        gridTemp[loopX,loopY,voxelsINSIDE] = 1

                elif gridCOzCROSS.size > 0:
                    # Remaining rays which meet the mesh in some way are not voxelised, but are labelled for correction later.
                    correctionLIST = np.insert( correctionLIST, correctionLIST.shape[0], [[loopX,loopY]], axis=0 )
    return gridTemp


if __name__=="__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    datapath="../modeldata"
    datapath = os.path.abspath(datapath)+"/"
    filename = os.path.join(datapath,"test.stl")
    meshVertexs = Read_stl(filename)[0]

    import time
    mvoxel = Voxel(120,120,120)
    start=time.time()
    mvoxel.VOXELISE(meshVertexs,"xyz",True)
    print('Python Voxelise time(P): {:6f}'.format(time.time()-start))

    mvoxel2 = Voxel(120,120,120)
    start=time.time()
    mvoxel2.VOXELISE(meshVertexs,"xyz",False)
    print('Python Voxelise time(S): {:6f}'.format(time.time()-start))

