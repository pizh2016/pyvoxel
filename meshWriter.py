import numpy as np
import base64

VTK_TO_ABAQUS_TYPE = {
    8: "S4R",
    9: "S4R",
    10: "C3D4",
    11: "C3D8R",
    12: "C3D8R"
}

def write_inp(FILENAME:str, type:int, nodes:np.ndarray, elements:np.ndarray):
    ele_offset = 1
    with open(FILENAME, "wt") as f:
        f.write("*HEADING\n")
        f.write('\n*Node\n')
        fmt = ", ".join(["{}"] + ["{:.9f}"] * nodes.shape[1]) + "\n"
        for k, x in enumerate(nodes):
            f.write(fmt.format(k + 1, *x))
        
        ele_nod = elements + ele_offset
        f.write('\n*Element, type={}\n'.format(VTK_TO_ABAQUS_TYPE[type]))

        for e in range(1, elements.shape[0] + 1):
            f.write('{:d},  {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}\n'.format(e,*ele_nod[e-1,:]))
    return



def write_vtk(FILENAME:str, type:int, nodes:np.ndarray, elements:np.ndarray):
    with open(FILENAME, "wt") as f:
        f.write(
            "# vtk DataFile Version 5.1\n"
            "Volume Mesh\n"
            "ASCII\n"
            "DATASET UNSTRUCTURED_GRID\n")

        f.write("POINTS  {:d}  float\n".format(nodes.shape[0]) )
        nodes.tofile(f, sep=" ")
        f.write("\n")

        f.write("CELLS {:d} {:d}\n".format(elements.shape[0]+1, elements.size))
        f.write("OFFSETS vtktypeint64\n")
        offsets = np.arange(0, elements.size+1, elements.shape[1], dtype=int)
        offsets.tofile(f, sep="\n")
        f.write("\n")

        f.write("CONNECTIVITY vtktypeint64\n")
        elements.tofile(f, sep="\n")
        f.write("\n")

        f.write("CELL_TYPES  {:d}\n".format(elements.shape[0]))
        np.full(elements.shape[0], type).tofile(f, sep="\n")
        f.write("\n")
    return



def write_vtu(FILENAME:str, type:int, nodes:np.ndarray, elements:np.ndarray):
    with open(FILENAME, "wt") as f:
        f.write(
            "<?xml version = \"1.0\"?>\n"
            "<VTKFile type = \"UnstructuredGrid\" version = \"0.1\" byte_order = \"LittleEndian\">\n"
            "<UnstructuredGrid>\n"
            "<Piece NumberOfPoints = \"{:d}\" NumberOfCells = \"{:d}\"> \n".format(nodes.shape[0], elements.shape[0])  )

        f.write(
            "<Points>\n"
            "<DataArray type = \"Float32\" Name = \"Points\" NumberOfComponents = \"3\" format = \"binary\">\n")
        data_bytes = nodes.astype(np.float32).tobytes()
        header = np.array(len(data_bytes), dtype=np.uint)
        f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

        f.write("\n"
            "</DataArray>\n"
            "</Points>\n"
            "<Cells>\n"
            "<DataArray type = \"Int32\" Name = \"connectivity\" format = \"binary\">\n")
        data_bytes = elements.astype(np.int32).tobytes()
        header = np.array(len(data_bytes), dtype=int)
        f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

        f.write("\n"
            "</DataArray>\n"
            "<DataArray type = \"Int32\" Name = \"offsets\" format = \"binary\">\n")
        offsets = np.arange(elements.shape[1], elements.size+1, elements.shape[1], dtype=np.int32)
        data_bytes = offsets.tobytes()
        header = np.array(len(data_bytes), dtype=int)
        f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

        f.write("\n"
            "</DataArray>\n"
            "<DataArray type = \"Int32\" Name = \"types\" format = \"binary\">\n")
        types = np.full(elements.shape[0], type).astype(np.int32)
        data_bytes = types.tobytes()
        header = np.array(len(data_bytes), dtype=int)
        f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

        f.write("\n"
            "</DataArray>\n"
            "</Cells>\n"
            "</Piece>\n"
            "</UnstructuredGrid>\n"
            "</VTKFile>\n")
    return



if __name__=='__main__':
    print("# vtk DataFile Version 5.1\n"
        "Volume Mesh\n"
        "ASCII\n"
        "DATASET UNSTRUCTURED_GRID\n")
    data_bytes = np.full(100,33333).tobytes()
    header = np.array(len(data_bytes), dtype=np.float64)
    print(int(33333).to_bytes(length=4,byteorder='little'))
    print(header.tobytes())
    print(base64.b64encode(header.tobytes() + data_bytes).decode())