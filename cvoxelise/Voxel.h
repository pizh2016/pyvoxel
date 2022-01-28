#pragma once
#include <stdio.h>
#include <algorithm>
#include <Eigen/Dense>
#include "MeshReader.h"
#include "MeshWriter.h"

#define PYPRINT 1;

using namespace Eigen;
template<typename T> using RowArrayX = Array<T, 1, Dynamic, RowMajor>;

template<class T>
RowArrayX<T> toRowArray(py::array_t<T> inarray) {
	py::buffer_info buffer_info = inarray.request();
	// extract data an shape of input array
	T* data = static_cast<T*>(buffer_info.ptr);
	std::vector<ssize_t> shape = buffer_info.shape;
	// wrap ndarray in Eigen::Map:
	Eigen::Map<RowArrayX<T>> array(data, buffer_info.size);
	return array;
}

// find where array value == 0
template<typename T> ssize_t argfind(py::array_t<T> arr) {
	for (ssize_t d = 0; d < arr.size(); d++)
		if (abs(arr.at(d)) <= 1e-8) return d;
	return -1;
}

template<typename T> py::array_t<T> setConstValue(py::array_t<T> arr, T arg) {
	py::buffer_info buf = arr.request();
	T* ptr = (T*)buf.ptr;
#pragma omp parallel for
	for (int i = 0; i < buf.size; i++)
		ptr[i] = arg;
	return arr;
}


class Voxel
{
public:
	Voxel();
	Voxel(int, int, int);
	int NX, NY, NZ;
	py::array_t<double> gridCOx, gridCOy, gridCOz;
	py::array_t<int> voxelgrid;
	ssize_t nnod, nele;
	py::array_t<int> ele_nod, nod_coor;
	py::array_t<double> nod_coor_abs;


	void initSize(int, int, int);
	py::array_t<int> VOXELISE(std::string, std::string ray = "xyz", bool parallel = true);
	py::array_t<int> VOXELISE(py::array_t<double>, std::string ray = "xyz", bool parallel = true);
	bool gen_vox_info();
	void save_mesh(std::string);

private:
	double __meshXmin, __meshYmin, __meshZmin, __meshXmax, __meshYmax, __meshZmax;
	py::array_t<int> _VOXELISE(py::array_t<double> , py::array_t<double> , py::array_t<double>, py::array_t<double>);
	py::array_t<int> _VOXELISE_Parallel(py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>);

};





