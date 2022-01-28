# Voxelise a mesh with python and cxx module

## Main functional modules


- The grid IO processing
  - `meshWriter.py`
    - `meshWriter.write_inp(FILENAME, type, nodes, elements)`
    - `meshWriter.write_vtk(FILENAME, type, nodes, elements)`
    - `meshWriter.write_vtu(FILENAME, type, nodes, elements)`
  - `pvoxel.py` 
    - `pvoxel.Read_stl(filename)`
    - `pvoxel.stlGetFormat(filename)`
- Python Voxelization function
  - `pvoxel.py`
    - `pvoxel.Voxel()`
- Call C++ voxel module
  - `voxel.py`
    - `voxel.Voxel()`

## Prerequisites

- Python **3** 
- numpy
- time
- multiprocessing
- joblib
- os
- sys

## Usage

```python
Python pvoxel
    import pvoxel
    meshVertexs = pvoxel.Read_stl(filename)[0]
    pvox = pvoxel.Voxel(NX,NY,NZ)
    pvox.VOXELISE(meshVertexs,ray='xyz',parallel=False)
    or pvox.VOXELISE(filename,ray='xyz',parallel=False)

    pvox.gen_vox_info()
    pvox.save_mesh(meshfilename)
```
```python
C++ voxel
    import voxel
    meshVertexs = cvoxel.Read_stl(filename)[0]
    cvox = voxel.Voxel(NX,NY,NZ)
    cvox.VOXELISE(meshVertexs,ray='xyz',parallel=True)
    or cvox.VOXELISE(filename,ray='xyz',parallel=True)

    cvox.gen_vox_info()
    cvox.save_mesh(meshfilename)
```

## Cxx module speed up Usage
You need to compile the cvoxelise module with Visual Studio (Versions above 2017), using cvoxelise.vcxproj.

#### dependency

- pybind11 
- python3 (>3.6)
- eigen3 
- openmp
#### config
Put the dynamic link library file  [cvoxelise.pyd] into the same directory as vovel.py