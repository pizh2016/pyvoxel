import time
import os
import voxel
import pvoxel


if __name__=='__main__':

    datapath="../modeldata"
    datapath = os.path.abspath(datapath)+"/"
    modellist = [x[:-4] for x in os.listdir(datapath) if os.path.isfile(datapath+x) and x.endswith(".stl")]
    model = "model"
    voxwidth = 4.0

    print('{0}{1}.stl'.format(datapath,model))
        
    meshVertexs = pvoxel.Read_stl('{0}{1}.stl'.format(datapath,model))[0]
    meshXmin = meshVertexs[:,0,:].min()
    meshXmax = meshVertexs[:,0,:].max()
    meshYmin = meshVertexs[:,1,:].min()
    meshYmax = meshVertexs[:,1,:].max()
    meshZmin = meshVertexs[:,2,:].min()
    meshZmax = meshVertexs[:,2,:].max()
        
    NX = int((meshXmax-meshXmin)/voxwidth+0.5)
    NY = int((meshYmax-meshYmin)/voxwidth+0.5)
    NZ = int((meshZmax-meshZmin)/voxwidth+0.5)
    print("Size: {:3f}  NX:{:d}  NY:{:d}  NZ:{:d}".format(voxwidth,NX,NY,NZ))

    cvox = voxel.Voxel(NX,NY,NZ)
    start=time.time()
    print("cvoxelEL init")
    cvox.VOXELISE(meshVertexs,ray='xyz',parallel=True)
    print('gridOUTPUT\n',cvox.voxelgrid.shape, cvox.voxelgrid.sum())
    print('C++ Voxelise time(P): {:6f}'.format(time.time()-start))

    start=time.time()
    cvox.gen_vox_info()
    # print("nod_coor_abs\n",cvox.nod_coor_abs)
    print('C++ Numbering time: {:6f}'.format(time.time()-start))
    
    start=time.time()
    cvox.save_mesh('{0}{1}.vtk'.format(datapath,model))
    print("Save vtk file FINISH!!!")
    print('C++ writing time: {:6f}'.format(time.time()-start))

    start=time.time()
    cvox.save_mesh('{0}{1}.vtu'.format(datapath,model))
    print("Save vtu file FINISH!!!")
    print('C++ writing time: {:6f}'.format(time.time()-start))

    start=time.time()
    cvox.save_mesh('{0}{1}.inp'.format(datapath,model))
    print("Save inp file FINISH!!!")
    print('C++ writing time: {:6f}'.format(time.time()-start))



    # pvox = pvoxel.Voxel(NX,NY,NZ)
    # start=time.time()
    # print("pvoxel init")
    # pvox.VOXELISE(meshVertexs,ray='xyz',parallel=False)
    # print('gridOUTPUT\n',pvox.voxelgrid.shape, pvox.voxelgrid.sum())
    # print('Python Voxelise time(P): {:6f}'.format(time.time()-start))

    # start=time.time()
    # pvox.gen_vox_info()
    # # print("nod_coor_abs\n",pvox.nod_coor_abs)
    # print('Python Numbering time: {:6f}'.format(time.time()-start))
    
    # start=time.time()
    # pvox.save_mesh('{0}{1}p.vtu'.format(datapath,model))
    # print("Save vtu file FINISH!!!")
    # print('Python writing time: {:6f}'.format(time.time()-start))

    # start=time.time()
    # pvox.save_mesh('{0}{1}p.vtk'.format(datapath,model))
    # print("Save vtk file FINISH!!!")
    # print('Python writing time: {:6f}'.format(time.time()-start))