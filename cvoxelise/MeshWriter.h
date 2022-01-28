#pragma once
#include <map>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include "base64.h"

namespace py = pybind11;
using namespace py::literals;
using namespace Eigen;

const std::map<int, std::string> VTK_TO_ABAQUS_TYPE = {
	{8, "S4R"},{9, "S4R"},{10, "C3D4"},{11, "C3D8R"},{12, "C3D8R"}
};
void write_inp(std::string FILENAME, int type, py::array_t<double> nodes, py::array_t<int> elements);
void write_vtk(std::string FILENAME, int tpye, py::array_t<double> nodes, py::array_t<int> elements);
void write_vtu(std::string FILENAME, int type, py::array_t<double> nodes, py::array_t<int> elements);


