#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>

// class STLInfo
// stlFILENAME   - String - Mandatory - The filename of the STL file.
// stlFORMAT - String - Optional - The format of the STL file :"ascii" or "binary"
// coordVERTEXS - Nx3x3 array - Mandatory
//                             - An array defining the vertex positions
//                              for each of the N facets, with :
//                              1 row for each facet
//                              3 cols for the x, y, z coordinates
//                              3 pages for the three VERTEXS
//
// coordNORMALS - Nx3 array - Optional
//                              -An array defining the normal vector for
//                              each of the N facets, with:
//                              1 row for each facet
//                              3 cols for the x, y, z components of the vector
//
// stlNAME - String - Optional - The name of the STL object
//

namespace py = pybind11;
using namespace py::literals;
//extern py::object np;


typedef struct STLInfo {
    std::string stlFILENAME;
    std::string stlFORMAT;
    std::string stlNAME;
    py::array_t<double> coordNORMALS;
    py::array_t<double> coordVERTEXS;
}stlInfo;

std::string stlGetFormat(std::string);
py::list Read_stl(std::string);
void READ_stlascii(stlInfo*);
void READ_stlbinary(stlInfo*);



