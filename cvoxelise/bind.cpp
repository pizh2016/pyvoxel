#include <pybind11/pybind11.h>
#include "MeshReader.h"
#include "Voxel.h"

//warper of the func 
PYBIND11_MODULE(cvoxelise, m) {
    m.doc() = "pybind11 cvoxelise plugin"; // optional module docstring
    py::class_<Voxel>(m, "Voxel")
        .def(py::init<>())
        .def(py::init<int, int, int>())
        .def_readonly("voxelgrid", &Voxel::voxelgrid)
        .def_readonly("gridCOx", &Voxel::gridCOx)
        .def_readonly("gridCOy", &Voxel::gridCOy)
        .def_readonly("gridCOz", &Voxel::gridCOz)
        .def_readonly("nnod", &Voxel::nnod)
        .def_readonly("nele", &Voxel::nele)
        .def_readonly("ele_nod", &Voxel::ele_nod)
        .def_readonly("nod_coor", &Voxel::nod_coor)
        .def_readonly("nod_coor_abs", &Voxel::nod_coor_abs)
        .def("initSize", &Voxel::initSize, "init size of X,Y Z")
        .def("VOXELISE", py::overload_cast<std::string, std::string, bool>(&Voxel::VOXELISE),
            pybind11::arg("filename"),
            pybind11::arg("raydirection") = "xyz",
            pybind11::arg("parallel") = true,
            "VOXELISE a stl model")
        .def("VOXELISE", py::overload_cast<py::array_t<double>, std::string, bool>(&Voxel::VOXELISE),
            pybind11::arg("mesh"),
            pybind11::arg("raydirection") = "xyz",
            pybind11::arg("parallel") = true,
            "VOXELISE a stl model")
        .def("gen_vox_info", &Voxel::gen_vox_info, "gen_vox_info of voxelgrid")
        .def("save_mesh",&Voxel::save_mesh, "Save voxel mesh to inp_file");
    m.def("Read_stl", &Read_stl, "Read stl file and covert to VERTEXS arrays list");
    m.def("stlGetFormat", &stlGetFormat, "Get stl file format : ascii or binary");
}