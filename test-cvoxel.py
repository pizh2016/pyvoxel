import time
import os
import voxel


if __name__=='__main__':

    datapath="../modeldata/ModelNew"
    datapath = os.path.abspath(datapath)+"/"
    modellist = [x[:-4] for x in os.listdir(datapath) if os.path.isfile(datapath+x) and x.endswith(".stl")]

    for model in modellist:
        print('{0}{1}.stl'.format(datapath,model))
        
        meshVertexs = voxel.Read_stl('{0}{1}.stl'.format(datapath,model))[0]
        voxwidth = 0.333333
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

        cvoxel = voxel.Voxel(NX,NY,NZ)
        start=time.time()
        gridOUTPUT = cvoxel.VOXELISE(meshVertexs,"xyz",True)
        print('gridOUTPUT\n',gridOUTPUT.sum())
        print('Voxelise time: {:6f}'.format(time.time()-start))

        start=time.time()
        cvoxel.gen_vox_info()
        # print(cvoxel.nod_coor_abs)
        print('Numbering time: {:6f}'.format(time.time()-start))

        start=time.time()
        cvoxel.save_mesh('{0}{1}.vtk'.format(datapath,model))
        print('Save file time: {:6f}'.format(time.time()-start))
