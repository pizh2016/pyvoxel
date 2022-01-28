import numpy as np

def test(NX,NY,NZ):
    
    n = int(NX)*int(NY)*int(NZ)
    nod_y, nod_x, nod_z = np.meshgrid(range(NX), range(NY), range(NZ))
     
    print(np.shape(nod_x))
    
    nod_coor = np.zeros((n, 3))
    nod_coor[:,0] = nod_x.flatten()
    nod_coor[:,1] = nod_y.flatten()
    nod_coor[:,2] = nod_z.T.flatten()
    print(nod_coor)
    return

def test_accurate(NX,NY,NZ):
    n = int(NX)*int(NY)*int(NZ)
    nod_coor = np.zeros((n, 3))
    order = 0
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                nod_coor[order,:] = [i,j,k]
                order += 1

    print(nod_coor)
    return

def mesh():
    import meshio
    # two triangles and one quad
    points = [
        [0.0, 0.0, 0],
        [1.0, 0.0, 0],
        [0.0, 1.0, 0],
        [1.0, 1.0, 0],
        [2.0, 0.0, 0],
        [2.0, 1.0, 0],
    ]
    cells = [
        ("triangle", [[0, 1, 2], [1, 3, 2]]),
        ("quad", [[1, 4, 5, 3]]),
    ]

    mesh = meshio.Mesh(
        points,
        cells,
        # Optionally provide extra data on points, cells, etc.
        point_data={"T": [0.3, -1.2, 0.5, 0.7, 0.0, -3.0],
                    "U": np.ones((6,3))},
        # Each item in cell data must match the cells array
        cell_data={"a": [[0.1, 0.2], [0.4]]},
    )
    mesh.cell_data["S Mises"]=[np.array([0.4, 0.2]),np.array([0.2])]
    print(mesh.cell_data)
    mesh.write("foo.vtu")

    # # Alternative with the same options
    # meshio.write_points_cells(
    #     "foo.vtu", 
    #     points, 
    #     cells
    #     )


if __name__=='__main__':

    test(3,4,5)
    # test_accurate(3,4,5)
    mesh()
