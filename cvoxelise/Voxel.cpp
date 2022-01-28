#include "Voxel.h"
#include <iomanip>

Voxel::Voxel() :nnod(0), nele(0){
	initSize(0, 0, 0);
}
Voxel::Voxel(int X, int Y, int Z) : nnod(0), nele(0){
	initSize(X, Y, Z);
}

void Voxel::initSize(int x, int y, int z)
{
	NX = x;
	NY = y;
	NZ = z;
}

py::array_t<int> Voxel::VOXELISE(std::string filename, std::string raydirection, bool parallel)
{
#if PYPRINT
	py::print("recieved mesh data form C++ stl-reader");
#endif // PYPRINTs
	py::array_t<double> meshXYZ = Read_stl(filename).attr("pop")(0);
	return VOXELISE(meshXYZ, raydirection, parallel);
}

py::array_t<int> Voxel::VOXELISE(py::array_t<double> mesh, std::string raydirection, bool parallel)
{
	
	py::array_t<double> meshXYZ = mesh;
	auto local = py::dict();
	// IDENTIFY THE MIN AND MAX X,Y,Z COORDINATES OF THE POLYGON MESH
	ssize_t N = meshXYZ.shape(0);
	//double meshXmin, meshYmin, meshZmin, meshXmax, meshYmax, meshZmax;
	__meshXmin = meshXYZ[py::make_tuple(py::slice(0, N, 1), 0, py::slice(0, 3, 1))].attr("min")().cast<double>();
	__meshXmax = meshXYZ[py::make_tuple(py::slice(0, N, 1), 0, py::slice(0, 3, 1))].attr("max")().cast<double>();
	__meshYmin = meshXYZ[py::make_tuple(py::slice(0, N, 1), 1, py::slice(0, 3, 1))].attr("min")().cast<double>();
	__meshYmax = meshXYZ[py::make_tuple(py::slice(0, N, 1), 1, py::slice(0, 3, 1))].attr("max")().cast<double>();
	__meshZmin = meshXYZ[py::make_tuple(py::slice(0, N, 1), 2, py::slice(0, 3, 1))].attr("min")().cast<double>();
	__meshZmax = meshXYZ[py::make_tuple(py::slice(0, N, 1), 2, py::slice(0, 3, 1))].attr("max")().cast<double>();

	double dx = (__meshXmax - __meshXmin) / (NX + 0.5);
	double dy = (__meshYmax - __meshYmin) / (NY + 0.5);
	double dz = (__meshZmax - __meshZmin) / (NZ + 0.5);

	RowVectorXd lx = RowVectorXd::LinSpaced(NX, __meshXmin + dx / 2.0, (__meshXmin + dx / 2.0) + dx * (NX - 1));
	RowVectorXd ly = RowVectorXd::LinSpaced(NY, __meshYmin + dy / 2.0, (__meshYmin + dy / 2.0) + dy * (NY - 1));
	RowVectorXd lz = RowVectorXd::LinSpaced(NZ, __meshZmin + dz / 2.0, (__meshZmin + dz / 2.0) + dz * (NZ - 1));


	//Check that the output grid is large enough to cover the mesh
	int gridcheckX = 0, gridcheckY = 0, gridcheckZ = 0;
	if (lx.minCoeff() > __meshXmin || lx.maxCoeff() < __meshXmax)
	{
		if (lx.minCoeff() > __meshXmin) gridcheckX = gridcheckX + 1;
		if (lx.maxCoeff() < __meshXmax) gridcheckX = gridcheckX + 2;
	}
	if (ly.minCoeff() > __meshYmin || ly.maxCoeff() < __meshYmax)
	{
		if (ly.minCoeff() > __meshYmin) gridcheckY = gridcheckY + 1;
		if (ly.maxCoeff() < __meshYmax) gridcheckY = gridcheckY + 2;
	}
	if (lz.minCoeff() > __meshZmin || lz.maxCoeff() < __meshZmax)
	{
		if (lz.minCoeff() > __meshZmin) gridcheckZ = gridcheckZ + 1;
		if (lz.maxCoeff() < __meshZmax) gridcheckZ = gridcheckZ + 2;
	}

	// copy Eigen::RowVectorXd to py::array_t
	switch (gridcheckX)
	{
	case 1:
		gridCOx = py::array_t<double>(NX + 1);
		gridCOx.mutable_data(0)[0] = __meshXmin;
		for (size_t i = 0; i < NX; i++)
			gridCOx.mutable_data(i + 1)[0] = lx(i);
		break;
	case 2:
		gridCOx = py::array_t<double>(NX + 1);
		for (size_t i = 0; i < NX; i++)
			gridCOx.mutable_data(i)[0] = lx(i);
		gridCOx.mutable_data(NX)[0] = __meshXmax;
		break;
	case 3:
		gridCOx = py::array_t<double>(NX + 2);
		gridCOx.mutable_data(0)[0] = __meshXmin;
		for (size_t i = 0; i < NX; i++)
			gridCOx.mutable_data(i + 1)[0] = lx(i);
		gridCOx.mutable_data(NX+1)[0] = __meshXmax;
		break;
	}
	switch (gridcheckY)
	{
	case 1:
		gridCOy = py::array_t<double>(NY + 1);
		gridCOy.mutable_data(0)[0] = __meshYmin;
		for (size_t i = 0; i < NY; i++)
			gridCOy.mutable_data(i + 1)[0] = ly(i);
		break;
	case 2:
		gridCOy = py::array_t<double>(NY + 1);
		for (size_t i = 0; i < NY; i++)
			gridCOy.mutable_data(i)[0] = ly(i);
		gridCOy.mutable_data(NY)[0] = __meshYmax;
		break;
	case 3:
		gridCOy = py::array_t<double>(NY + 2);
		gridCOy.mutable_data(0)[0] = __meshYmin;
		for (size_t i = 0; i < NY; i++)
			gridCOy.mutable_data(i + 1)[0] = ly(i);
		gridCOy.mutable_data(NY + 1)[0] = __meshYmax;
		break;
	}
	switch (gridcheckZ)
	{
	case 1:
		gridCOz = py::array_t<double>(NZ + 1);
		gridCOz.mutable_data(0)[0] = __meshZmin;
		for (size_t i = 0; i < NZ; i++)
			gridCOz.mutable_data(i + 1)[0] = lz(i);
		break;
	case 2:
		gridCOz = py::array_t<double>(NZ + 1);
		for (size_t i = 0; i < NZ; i++)
			gridCOz.mutable_data(i)[0] = lz(i);
		gridCOz.mutable_data(NZ)[0] = __meshZmax;
		break;
	case 3:
		gridCOz = py::array_t<double>(NZ + 2);
		gridCOz.mutable_data(0)[0] = __meshZmin;
		for (size_t i = 0; i < NZ; i++)
			gridCOz.mutable_data(i + 1)[0] = lz(i);
		gridCOz.mutable_data(NZ+1)[0] = __meshZmax;
		break;
	}

	
	// VOXELISE USING THE USER DEFINED RAY DIRECTION
	//======================================================

	// Count the number of voxels in each direction :
	size_t voxcountX = gridCOx.size();
	size_t voxcountY = gridCOy.size();
	size_t voxcountZ = gridCOz.size();
	size_t raysize = raydirection.size();
	// Prepare logical array to hold the voxelised data :
	py::array_t<int> gridOUTPUT = py::array_t<int>(voxcountX*voxcountY*voxcountZ*raysize).attr("reshape")(std::make_tuple(voxcountX,voxcountY,voxcountZ,raysize));
	setConstValue(gridOUTPUT, 0);
#ifdef PYPRINT
	py::print("gridOUTPUT Init: ", "(",gridOUTPUT.shape(0), gridOUTPUT.shape(1), gridOUTPUT.shape(2), gridOUTPUT.shape(3), ")", gridOUTPUT.attr("sum")());
#endif // PYPRINT

	int countdirections = 0;
	if (raydirection.find("x") + 1) {
		auto slice = py::make_tuple(py::slice(0, N, 1), py::make_tuple(1, 2, 0), py::slice(0, 3, 1));
		py::array_t<int> rayX;
		if(parallel) rayX = _VOXELISE_Parallel(gridCOy, gridCOz, gridCOx, meshXYZ[slice].cast<py::array_t<double>>());
		else rayX = _VOXELISE(gridCOy, gridCOz, gridCOx, meshXYZ[slice].cast<py::array_t<double>>());
		gridOUTPUT[py::make_tuple(py::slice(0, voxcountX, 1),
			py::slice(0, voxcountY, 1),
			py::slice(0, voxcountZ, 1),
			countdirections)] = rayX.attr("transpose")(py::make_tuple(2, 0, 1));
		countdirections = countdirections + 1;
#ifdef PYPRINT
		py::print("gridOUTPUT x: ", "(",gridOUTPUT.shape(0), gridOUTPUT.shape(1), gridOUTPUT.shape(2), gridOUTPUT.shape(3), ")", gridOUTPUT.attr("sum")());
#endif // PYPRINT
	}
	if (raydirection.find("y") + 1) {
		auto slice = py::make_tuple(py::slice(0, N, 1), py::make_tuple(2, 0, 1), py::slice(0, 3, 1));
		py::array_t<int> rayY;
		if(parallel) rayY = _VOXELISE_Parallel(gridCOz, gridCOx, gridCOy, meshXYZ[slice].cast<py::array_t<double>>());
		else rayY = _VOXELISE(gridCOz, gridCOx, gridCOy, meshXYZ[slice].cast<py::array_t<double>>());
		gridOUTPUT[py::make_tuple(py::slice(0, voxcountX, 1),
			py::slice(0, voxcountY, 1),
			py::slice(0, voxcountZ, 1),
			countdirections)] = rayY.attr("transpose")(py::make_tuple(1, 2, 0));
		countdirections = countdirections + 1;
#ifdef PYPRINT
		py::print("gridOUTPUT y: ", "(", gridOUTPUT.shape(0), gridOUTPUT.shape(1), gridOUTPUT.shape(2), gridOUTPUT.shape(3), ")", gridOUTPUT.attr("sum")());
#endif // PYPRINT
	}
	if (raydirection.find("z") + 1) {
		py::array_t<int> rayZ;
		if (parallel) rayZ = _VOXELISE_Parallel(gridCOx, gridCOy, gridCOz, meshXYZ);
		else rayZ = _VOXELISE(gridCOx, gridCOy, gridCOz, meshXYZ);
		gridOUTPUT[py::make_tuple(py::slice(0, voxcountX, 1),
			py::slice(0, voxcountY, 1),
			py::slice(0, voxcountZ, 1),
			countdirections)] = rayZ;
		countdirections = countdirections + 1;
#ifdef PYPRINT
		py::print("gridOUTPUT z: ", "(", gridOUTPUT.shape(0), gridOUTPUT.shape(1), gridOUTPUT.shape(2), gridOUTPUT.shape(3), ")", gridOUTPUT.attr("sum")());
#endif // PYPRINT
	}
	// Combine the results of each ray - tracing direction :
	if (raydirection.size() > 1) {
		local["gridOUT"] = gridOUTPUT.attr("sum")("axis"_a = 3);
		local["raySize"] = raydirection.size() / 2;
		gridOUTPUT = py::eval("gridOUT >= raySize", local);
	}

	// RETURN THE OUTPUT GRID TO THE SIZE REQUIRED BY THE USER (IF IT WAS CHANGED EARLIER)
	//======================================================
	switch (gridcheckX)
	{
	case 1:
		gridOUTPUT = gridOUTPUT[py::make_tuple(py::slice(1, voxcountX, 1), py::slice(0, voxcountY, 1), py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>();
		gridCOx = gridCOx[py::make_tuple(py::slice(1, voxcountX, 1))].cast<py::array_t<double>>();
		break;
	case 2:
		gridOUTPUT = gridOUTPUT[py::make_tuple(py::slice(0, voxcountX - 1, 1), py::slice(0, voxcountY, 1), py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>();
		gridCOx = gridCOx[py::make_tuple(py::slice(0, voxcountX - 1, 1))].cast<py::array_t<double>>();
		break;
	case 3:
		gridOUTPUT = gridOUTPUT[py::make_tuple(py::slice(1, voxcountX - 1, 1), py::slice(0, voxcountY, 1), py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>();
		gridCOx = gridCOx[py::make_tuple(py::slice(1, voxcountX - 1, 1))].cast<py::array_t<double>>();
		break;
	}

	switch (gridcheckY)
	{
	case 1:
		gridOUTPUT = gridOUTPUT[py::make_tuple(py::slice(0, voxcountX, 1), py::slice(1, voxcountY, 1), py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>();
		gridCOy = gridCOy[py::make_tuple(py::slice(1, voxcountY, 1))].cast<py::array_t<double>>();
		break;
	case 2:
		gridOUTPUT = gridOUTPUT[py::make_tuple(py::slice(0, voxcountX, 1), py::slice(0, voxcountY - 1, 1), py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>();
		gridCOy = gridCOy[py::make_tuple(py::slice(0, voxcountY - 1, 1))].cast<py::array_t<double>>();
		break;
	case 3:
		gridOUTPUT = gridOUTPUT[py::make_tuple(py::slice(0, voxcountX, 1), py::slice(1, voxcountY - 1, 1), py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>();
		gridCOy = gridCOy[py::make_tuple(py::slice(1, voxcountY - 1, 1))].cast<py::array_t<double>>();
		break;
	}

	switch (gridcheckZ)
	{
	case 1:
		gridOUTPUT = gridOUTPUT[py::make_tuple(py::slice(0, voxcountX, 1), py::slice(0, voxcountY, 1), py::slice(1, voxcountZ, 1))].cast<py::array_t<int>>();
		gridCOz = gridCOz[py::make_tuple(py::slice(1, voxcountZ, 1))].cast<py::array_t<double>>();
		break;
	case 2:
		gridOUTPUT = gridOUTPUT[py::make_tuple(py::slice(0, voxcountX, 1), py::slice(0, voxcountY, 1), py::slice(0, voxcountZ - 1, 1))].cast<py::array_t<int>>();
		gridCOz = gridCOz[py::make_tuple(py::slice(0, voxcountZ - 1, 1))].cast<py::array_t<double>>();
		break;
	case 3:
		gridOUTPUT = gridOUTPUT[py::make_tuple(py::slice(0, voxcountX, 1), py::slice(0, voxcountY, 1), py::slice(1, voxcountZ - 1, 1))].cast<py::array_t<int>>();
		gridCOz = gridCOz[py::make_tuple(py::slice(1, voxcountZ - 1, 1))].cast<py::array_t<double>>();
		break;
	}
#ifdef PYPRINT
	py::print("voxelgrid sum: ", "(", gridOUTPUT.shape(0), gridOUTPUT.shape(1), gridOUTPUT.shape(2), ")", gridOUTPUT.attr("sum")());
#endif // PYPRINT
	voxelgrid = gridOUTPUT;
	return voxelgrid;
}


py::array_t<int> Voxel::_VOXELISE(py::array_t<double> gx, py::array_t<double> gy, py::array_t<double> gz, py::array_t<double> meshXYZ)
{
	// get_eigen_array from numpy ndarray
	RowArrayX<double> gridCOx = toRowArray(gx);
	RowArrayX<double> gridCOy = toRowArray(gy);
	RowArrayX<double> gridCOz = toRowArray(gz);
	// Count the number of voxels in each direction :
	ssize_t voxcountX = gridCOx.size();
	ssize_t voxcountY = gridCOy.size();
	ssize_t voxcountZ = gridCOz.size();
	// Prepare logical array to hold the voxelised data :
	py::array_t<int> gridOUTPUT = py::array_t<int>(voxcountX * voxcountY * voxcountZ).attr("reshape")(std::make_tuple(voxcountX, voxcountY, voxcountZ));
	setConstValue(gridOUTPUT, (int)0);

	ssize_t N = meshXYZ.shape(0);
	double meshXmin, meshXmax, meshYmin, meshYmax, meshZmin, meshZmax;
	meshXmin = meshXYZ[py::make_tuple(py::slice(0, N, 1), 0, py::slice(0, 3, 1))].attr("min")().cast<double>();
	meshXmax = meshXYZ[py::make_tuple(py::slice(0, N, 1), 0, py::slice(0, 3, 1))].attr("max")().cast<double>();
	meshYmin = meshXYZ[py::make_tuple(py::slice(0, N, 1), 1, py::slice(0, 3, 1))].attr("min")().cast<double>();
	meshYmax = meshXYZ[py::make_tuple(py::slice(0, N, 1), 1, py::slice(0, 3, 1))].attr("max")().cast<double>();
	meshZmin = meshXYZ[py::make_tuple(py::slice(0, N, 1), 2, py::slice(0, 3, 1))].attr("min")().cast<double>();
	meshZmax = meshXYZ[py::make_tuple(py::slice(0, N, 1), 2, py::slice(0, 3, 1))].attr("max")().cast<double>();

	// Identify the min and max x, y coordinates(pixels) of the mesh :
	ssize_t meshXminp, meshXmaxp, meshYminp, meshYmaxp, temp, row;
	RowArrayX<double> gxmin = (gridCOx - meshXmin).abs();
	RowArrayX<double> gxmax = (gridCOx - meshXmax).abs();
	RowArrayX<double> gymin = (gridCOy - meshYmin).abs();
	RowArrayX<double> gymax = (gridCOy - meshYmax).abs();
	//find where gxmin == gxmin.min()
	gxmin.minCoeff(&row,&meshXminp);
	gxmax.minCoeff(&row,&meshXmaxp);	
	gymin.minCoeff(&row,&meshYminp);
	gymax.minCoeff(&row,&meshYmaxp);

	// Make sure min < max for the mesh coordinates :
	if (meshXminp > meshXmaxp) {
		temp = meshXminp;
		meshXminp = meshXmaxp;
		meshXmaxp = temp;
	}
	if (meshYminp > meshYmaxp) {
		temp = meshYminp;
		meshYminp = meshYmaxp;
		meshYmaxp = temp;
	}
	// Identify the min and max x, y, z coordinates of each facet :
	py::array_t<double> meshXYZmin = meshXYZ.attr("min")("axis"_a = 2);
	py::array_t<double> meshXYZmax = meshXYZ.attr("max")("axis"_a = 2);

	// VOXELISE THE MESH
	//======================================================

	std::vector<int> correctionLIST; // shape N x 2 later
	//The mesh will be voxelised by passing rays in the z - direction through each x, y pixel, 
	for (int loopY = meshYminp; loopY < meshYmaxp + 1; loopY++) {
		// - 1a - Find which mesh facets could possibly be crossed by the ray:
		std::vector<int> possibleCROSSLISTy;
		for (int i = 0; i < meshXYZmin.shape(0); i++) {
			if ((meshXYZmin.mutable_at(i,1) <= gridCOy(loopY)) && (meshXYZmax.mutable_at(i, 1) >= gridCOy(loopY)))
				possibleCROSSLISTy.push_back(i);
		}

		for (int loopX = meshXminp; loopX < meshXmaxp + 1; loopX++) {
			// - 1b - Find which mesh facets could possibly be crossed by the ray:
			std::vector<int> possibleCROSSLIST;
			for (int j = 0; j < possibleCROSSLISTy.size(); j++) {
				if ((meshXYZmin.mutable_at(possibleCROSSLISTy.at(j), 0) <= gridCOx(loopX)) && (meshXYZmax.mutable_at(possibleCROSSLISTy.at(j), 0) >= gridCOx(loopX)))
					possibleCROSSLIST.push_back(possibleCROSSLISTy.at(j));
			}

			if (possibleCROSSLIST.size() > 0) {
				// -2 - For each facet, check if the ray really does cross the facet rather than just passing it close - by:
				std::vector<int> facetCROSSLIST;
				// - 2 - Check for crossed facets
				//Only continue the analysis if some nearby facets were actually identified
				double Y1predicted, Y2predicted, Y3predicted, YRpredicted;
				double planecoA, planecoB, planecoC, planecoD;
				for (int loopCHECKFACET : possibleCROSSLIST) {
					//Check if ray crosses the facet.This method is much(>> 10 times) faster than using the built - in function 'inpolygon'.
					//Taking each edge of the facet in turn, check if the ray is on the same side as the opposing vertex.
					Y1predicted = meshXYZ.mutable_at(loopCHECKFACET, 1, 1) - ((meshXYZ.mutable_at(loopCHECKFACET, 1, 1) - meshXYZ.mutable_at(loopCHECKFACET, 1, 2)) * (meshXYZ.mutable_at(loopCHECKFACET, 0, 1) - meshXYZ.mutable_at(loopCHECKFACET, 0, 0)) / (meshXYZ.mutable_at(loopCHECKFACET, 0, 1) - meshXYZ.mutable_at(loopCHECKFACET, 0, 2)));
					YRpredicted = meshXYZ.mutable_at(loopCHECKFACET, 1, 1) - ((meshXYZ.mutable_at(loopCHECKFACET, 1, 1) - meshXYZ.mutable_at(loopCHECKFACET, 1, 2)) * (meshXYZ.mutable_at(loopCHECKFACET, 0, 1) - gridCOx(loopX)) / (meshXYZ.mutable_at(loopCHECKFACET, 0, 1) - meshXYZ.mutable_at(loopCHECKFACET, 0, 2)));

					if ((Y1predicted > meshXYZ.mutable_at(loopCHECKFACET, 1, 0) && YRpredicted > gridCOy(loopY)) || (Y1predicted < meshXYZ.mutable_at(loopCHECKFACET, 1, 0) && YRpredicted < gridCOy(loopY))) {
						//The ray is on the same side of the 2 - 3 edge as the 1st vertex.
						Y2predicted = meshXYZ.mutable_at(loopCHECKFACET, 1, 2) - ((meshXYZ.mutable_at(loopCHECKFACET, 1, 2) - meshXYZ.mutable_at(loopCHECKFACET, 1, 0)) * (meshXYZ.mutable_at(loopCHECKFACET, 0, 2) - meshXYZ.mutable_at(loopCHECKFACET, 0, 1)) / (meshXYZ.mutable_at(loopCHECKFACET, 0, 2) - meshXYZ.mutable_at(loopCHECKFACET, 0, 0)));
						YRpredicted = meshXYZ.mutable_at(loopCHECKFACET, 1, 2) - ((meshXYZ.mutable_at(loopCHECKFACET, 1, 2) - meshXYZ.mutable_at(loopCHECKFACET, 1, 0)) * (meshXYZ.mutable_at(loopCHECKFACET, 0, 2) - gridCOx(loopX)) / (meshXYZ.mutable_at(loopCHECKFACET, 0, 2) - meshXYZ.mutable_at(loopCHECKFACET, 0, 0)));

						if ((Y2predicted > meshXYZ.mutable_at(loopCHECKFACET, 1, 1) && YRpredicted > gridCOy(loopY)) || (Y2predicted < meshXYZ.mutable_at(loopCHECKFACET, 1, 1) && YRpredicted < gridCOy(loopY))) {
							//The ray is on the same side of the 3 - 1 edge as the 2nd vertex.
							Y3predicted = meshXYZ.mutable_at(loopCHECKFACET, 1, 0) - ((meshXYZ.mutable_at(loopCHECKFACET, 1, 0) - meshXYZ.mutable_at(loopCHECKFACET, 1, 1)) * (meshXYZ.mutable_at(loopCHECKFACET, 0, 0) - meshXYZ.mutable_at(loopCHECKFACET, 0, 2)) / (meshXYZ.mutable_at(loopCHECKFACET, 0, 0) - meshXYZ.mutable_at(loopCHECKFACET, 0, 1)));
							YRpredicted = meshXYZ.mutable_at(loopCHECKFACET, 1, 0) - ((meshXYZ.mutable_at(loopCHECKFACET, 1, 0) - meshXYZ.mutable_at(loopCHECKFACET, 1, 1)) * (meshXYZ.mutable_at(loopCHECKFACET, 0, 0) - gridCOx(loopX)) / (meshXYZ.mutable_at(loopCHECKFACET, 0, 0) - meshXYZ.mutable_at(loopCHECKFACET, 0, 1)));

							if ((Y3predicted > meshXYZ.mutable_at(loopCHECKFACET, 1, 2) && YRpredicted > gridCOy(loopY)) || (Y3predicted < meshXYZ.mutable_at(loopCHECKFACET, 1, 2) && YRpredicted < gridCOy(loopY)))
								facetCROSSLIST.push_back(loopCHECKFACET);

						}
					}
				}

				// - 3 - Find the z coordinate of the locations where the ray crosses each facet or vertex:
				RowArrayX<double> gridCOzCROSS(facetCROSSLIST.size());
				std::vector<double> insideL;
				for (int index = 0; index < facetCROSSLIST.size(); index++) {
					// Define the equation describing the plane of the facet
					//    Ax + By + Cz + D = 0
					//    where  A = y1(z2 - z3) + y2(z3 - z1) + y3(z1 - z2)
					//           B = z1(x2 - x3) + z2(x3 - x1) + z3(x1 - x2)
					//           C = x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)
					//           D = -x1(y2 z3 - y3 z2) - x2(y3 z1 - y1 z3) - x3(y1 z2 - y2 z1)
					// For the x and y coordinates of the ray, solve these equations to find the z coordinate in this plane.
					int loopFINDZ = facetCROSSLIST.at(index);
					planecoA = meshXYZ.mutable_at(loopFINDZ, 1, 0) * (meshXYZ.mutable_at(loopFINDZ, 2, 1) - meshXYZ.mutable_at(loopFINDZ, 2, 2)) + meshXYZ.mutable_at(loopFINDZ, 1, 1) * (meshXYZ.mutable_at(loopFINDZ, 2, 2) - meshXYZ.mutable_at(loopFINDZ, 2, 0)) + meshXYZ.mutable_at(loopFINDZ, 1, 2) * (meshXYZ.mutable_at(loopFINDZ, 2, 0) - meshXYZ.mutable_at(loopFINDZ, 2, 1));
					planecoB = meshXYZ.mutable_at(loopFINDZ, 2, 0) * (meshXYZ.mutable_at(loopFINDZ, 0, 1) - meshXYZ.mutable_at(loopFINDZ, 0, 2)) + meshXYZ.mutable_at(loopFINDZ, 2, 1) * (meshXYZ.mutable_at(loopFINDZ, 0, 2) - meshXYZ.mutable_at(loopFINDZ, 0, 0)) + meshXYZ.mutable_at(loopFINDZ, 2, 2) * (meshXYZ.mutable_at(loopFINDZ, 0, 0) - meshXYZ.mutable_at(loopFINDZ, 0, 1));
					planecoC = meshXYZ.mutable_at(loopFINDZ, 0, 0) * (meshXYZ.mutable_at(loopFINDZ, 1, 1) - meshXYZ.mutable_at(loopFINDZ, 1, 2)) + meshXYZ.mutable_at(loopFINDZ, 0, 1) * (meshXYZ.mutable_at(loopFINDZ, 1, 2) - meshXYZ.mutable_at(loopFINDZ, 1, 0)) + meshXYZ.mutable_at(loopFINDZ, 0, 2) * (meshXYZ.mutable_at(loopFINDZ, 1, 0) - meshXYZ.mutable_at(loopFINDZ, 1, 1));
					planecoD = -meshXYZ.mutable_at(loopFINDZ, 0, 0) * (meshXYZ.mutable_at(loopFINDZ, 1, 1) * meshXYZ.mutable_at(loopFINDZ, 2, 2) - meshXYZ.mutable_at(loopFINDZ, 1, 2) * meshXYZ.mutable_at(loopFINDZ, 2, 1)) - meshXYZ.mutable_at(loopFINDZ, 0, 1) * (meshXYZ.mutable_at(loopFINDZ, 1, 2) * meshXYZ.mutable_at(loopFINDZ, 2, 0) - meshXYZ.mutable_at(loopFINDZ, 1, 0) \
						* meshXYZ.mutable_at(loopFINDZ, 2, 2)) - meshXYZ.mutable_at(loopFINDZ, 0, 2) * (meshXYZ.mutable_at(loopFINDZ, 1, 0) * meshXYZ.mutable_at(loopFINDZ, 2, 1) - meshXYZ.mutable_at(loopFINDZ, 1, 1) * meshXYZ.mutable_at(loopFINDZ, 2, 0));

					if (abs(planecoC) < 1e-14)
						planecoC = 0.0;
					gridCOzCROSS(index) = (-planecoD - planecoA * gridCOx(loopX) - planecoB * gridCOy(loopY)) / planecoC;
					if ((gridCOzCROSS(index) >= meshZmin - 1e-12) && (gridCOzCROSS(index) <= meshZmax + 1e-12))
						insideL.push_back(gridCOzCROSS(index));
				}
				//Remove values of gridCOzCROSS which are outside of the mesh limits(including a 1e-12 margin for error).
				std::sort(insideL.begin(), insideL.end());
				gridCOzCROSS = Eigen::Map<RowArrayX<double>>(insideL.data(),insideL.size());
				//Round gridCOzCROSS to remove any rounding errors, and take only the unique values :
				gridCOzCROSS = (gridCOzCROSS * 1e12).round() / 1e12;
				int aa = std::unique(gridCOzCROSS.data(), gridCOzCROSS.data() + gridCOzCROSS.size()) - gridCOzCROSS.data();
				gridCOzCROSS = Map<RowArrayX<double>>(gridCOzCROSS.data(),aa);

				// - 4 - Label as being inside the mesh all the voxels that the ray passes through after crossing one facet before crossing another facet
				// Only rays which cross an even number of facets are voxelised

				if ((gridCOzCROSS.size() % 2) == 0) {
					for (int loopASSIGN = 1; loopASSIGN < (gridCOzCROSS.size() / 2.0) + 1; loopASSIGN++) {
						for (int sz = 0; sz < gridCOz.size(); sz++) {
							if ((gridCOz(sz) > gridCOzCROSS(2*loopASSIGN - 2)) && (gridCOz(sz) < gridCOzCROSS(2*loopASSIGN - 1)))
								gridOUTPUT.mutable_at(loopX, loopY, sz) = 1;
						}
					}
				}

				// Remaining rays which meet the mesh in some way are not voxelised, but are labelled for correction later.
				else if (gridCOzCROSS.size() > 0) {
					correctionLIST.push_back(loopX);
					correctionLIST.push_back(loopY);
				}

			}

		}


	}
	//For rays where the voxelisation did not give a clear result, the ray is computed by interpolating from the surrounding rays.
	ssize_t countCORRECTIONLIST = correctionLIST.size()/2;
	if (countCORRECTIONLIST > 0) {
		Map<Array<int, Dynamic, 2, RowMajor>> correctionArray (correctionLIST.data(), countCORRECTIONLIST, 2);
		//If necessary, add a one - pixel border around the x and y edges of thearray.
		//This prevents an error if the code tries to interpolate a ray at the edge of the x, y grid.
		if (correctionArray.colwise().minCoeff()(0) == 1 || correctionArray.colwise().maxCoeff()(0) == gridCOx.size() ||
			correctionArray.colwise().minCoeff()(1) == 1 ||correctionArray.colwise().maxCoeff()(1) == gridCOy.size()) {
			// padding in X,Y. not in Z
			py::array_t<int> gridpadding = py::array_t<int>((voxcountX+2) * (voxcountY+2) * voxcountZ).attr("reshape")(std::make_tuple(voxcountX+2, voxcountY+2, voxcountZ));
			setConstValue(gridpadding, (int)0);
			gridpadding[py::make_tuple(py::slice(1, voxcountX+1, 1),
				py::slice(1, voxcountY+1, 1),
				py::slice(0, voxcountZ, 1))] = gridOUTPUT;
			gridOUTPUT = gridpadding;
			correctionArray = correctionArray + 1;

		}
		for (int loopC = 0; loopC < countCORRECTIONLIST; loopC++) {
			int lx = correctionArray(loopC, 0);
			int ly = correctionArray(loopC, 1);
			// Surrounding elements
			// t1 t2 t3
			// t4    t5
			// t6 t7 t8
			RowArrayX<int> t1 = toRowArray(gridOUTPUT[py::make_tuple(lx - 1, ly - 1, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t2 = toRowArray(gridOUTPUT[py::make_tuple(lx - 1, ly, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t3 = toRowArray(gridOUTPUT[py::make_tuple(lx - 1, ly + 1, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t4 = toRowArray(gridOUTPUT[py::make_tuple(lx, ly - 1, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t5 = toRowArray(gridOUTPUT[py::make_tuple(lx, ly + 1, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t6 = toRowArray(gridOUTPUT[py::make_tuple(lx + 1, ly - 1, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t7 = toRowArray(gridOUTPUT[py::make_tuple(lx + 1, ly, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t8 = toRowArray(gridOUTPUT[py::make_tuple(lx + 1, ly + 1, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			Array<int,8,Dynamic,RowMajor> ttotal(8, t1.size());
			ttotal << t1,t2,t3,t4,t5,t6,t7,t8;

			RowArrayX<int> voxelsforcorrection = ttotal.colwise().sum();
			for (int ii = 0; ii < voxelsforcorrection.size(); ii++)
				if (voxelsforcorrection(ii) >= 4)
					gridOUTPUT.mutable_at(lx, ly, ii) = 1;
		}
		//Remove the one - pixel border surrounding the array, if this was added previously.
		if (gridOUTPUT.shape(0) > gridCOx.size() || gridOUTPUT.shape(1) > gridCOy.size())
			gridOUTPUT = gridOUTPUT[py::make_tuple(py::slice(1, voxcountX + 1, 1), py::slice(1, voxcountY + 1, 1), py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>();
	}
	return gridOUTPUT;
}

py::array_t<int> Voxel::_VOXELISE_Parallel(py::array_t<double> gx, py::array_t<double> gy, py::array_t<double> gz, py::array_t<double> meshXYZ)
{
	// get_eigen_array from numpy ndarray
	RowArrayX<double> gridCOx = toRowArray(gx);
	RowArrayX<double> gridCOy = toRowArray(gy);
	RowArrayX<double> gridCOz = toRowArray(gz);
	// Count the number of voxels in each direction :
	ssize_t voxcountX = gridCOx.size();
	ssize_t voxcountY = gridCOy.size();
	ssize_t voxcountZ = gridCOz.size();
	// Prepare logical array to hold the voxelised data :
	py::array_t<int> gridOUTPUT = py::array_t<int>(voxcountX * voxcountY * voxcountZ).attr("reshape")(std::make_tuple(voxcountX, voxcountY, voxcountZ));
	setConstValue(gridOUTPUT, (int)0);

	ssize_t N = meshXYZ.shape(0);
	double meshXmin, meshYmin, meshZmin, meshXmax, meshYmax, meshZmax;
	meshXmin = meshXYZ[py::make_tuple(py::slice(0, N, 1), 0, py::slice(0, 3, 1))].attr("min")().cast<double>();
	meshXmax = meshXYZ[py::make_tuple(py::slice(0, N, 1), 0, py::slice(0, 3, 1))].attr("max")().cast<double>();
	meshYmin = meshXYZ[py::make_tuple(py::slice(0, N, 1), 1, py::slice(0, 3, 1))].attr("min")().cast<double>();
	meshYmax = meshXYZ[py::make_tuple(py::slice(0, N, 1), 1, py::slice(0, 3, 1))].attr("max")().cast<double>();
	meshZmin = meshXYZ[py::make_tuple(py::slice(0, N, 1), 2, py::slice(0, 3, 1))].attr("min")().cast<double>();
	meshZmax = meshXYZ[py::make_tuple(py::slice(0, N, 1), 2, py::slice(0, 3, 1))].attr("max")().cast<double>();

	// Identify the min and max x, y coordinates(pixels) of the mesh :
	ssize_t meshXminp, meshXmaxp, meshYminp, meshYmaxp, temp, row;
	RowArrayX<double> gxmin = (gridCOx - meshXmin).abs();
	RowArrayX<double> gxmax = (gridCOx - meshXmax).abs();
	RowArrayX<double> gymin = (gridCOy - meshYmin).abs();
	RowArrayX<double> gymax = (gridCOy - meshYmax).abs();
	//find where gxmin == gxmin.min()
	gxmin.minCoeff(&row, &meshXminp);
	gxmax.minCoeff(&row, &meshXmaxp);
	gymin.minCoeff(&row, &meshYminp);
	gymax.minCoeff(&row, &meshYmaxp);

	// Make sure min < max for the mesh coordinates :
	if (meshXminp > meshXmaxp) {
		temp = meshXminp;
		meshXminp = meshXmaxp;
		meshXmaxp = temp;
	}
	if (meshYminp > meshYmaxp) {
		temp = meshYminp;
		meshYminp = meshYmaxp;
		meshYmaxp = temp;
	}
	// Identify the min and max x, y, z coordinates of each facet :
	py::array_t<double> meshXYZmin = meshXYZ.attr("min")("axis"_a = 2);
	py::array_t<double> meshXYZmax = meshXYZ.attr("max")("axis"_a = 2);

	// VOXELISE THE MESH
	//======================================================

	std::vector<int> correctionLIST; // shape N x 2 later
	//The mesh will be voxelised by passing rays in the z - direction through each x, y pixel, 
#pragma omp parallel for shared(correctionLIST)
	for (int loopY = meshYminp; loopY < meshYmaxp + 1; loopY++) {
		// - 1a - Find which mesh facets could possibly be crossed by the ray:
		std::vector<int> possibleCROSSLISTy;
		for (int i = 0; i < meshXYZmin.shape(0); i++) {
			if ((meshXYZmin.mutable_at(i, 1) <= gridCOy(loopY)) && (meshXYZmax.mutable_at(i, 1) >= gridCOy(loopY)))
				possibleCROSSLISTy.push_back(i);
		}

		for (int loopX = meshXminp; loopX < meshXmaxp + 1; loopX++) {
			// - 1b - Find which mesh facets could possibly be crossed by the ray:
			std::vector<int> possibleCROSSLIST;
			for (int j = 0; j < possibleCROSSLISTy.size(); j++) {
				if ((meshXYZmin.mutable_at(possibleCROSSLISTy.at(j), 0) <= gridCOx(loopX)) && (meshXYZmax.mutable_at(possibleCROSSLISTy.at(j), 0) >= gridCOx(loopX)))
					possibleCROSSLIST.push_back(possibleCROSSLISTy.at(j));
			}

			if (possibleCROSSLIST.size() > 0) {
				// -2 - For each facet, check if the ray really does cross the facet rather than just passing it close - by:
				std::vector<int> facetCROSSLIST;
				// - 2 - Check for crossed facets
				//Only continue the analysis if some nearby facets were actually identified
				double Y1predicted, Y2predicted, Y3predicted, YRpredicted;
				double planecoA, planecoB, planecoC, planecoD;
				for (int loopCHECKFACET : possibleCROSSLIST) {
					//Check if ray crosses the facet.This method is much(>> 10 times) faster than using the built - in function 'inpolygon'.
					//Taking each edge of the facet in turn, check if the ray is on the same side as the opposing vertex.
					Y1predicted = meshXYZ.mutable_at(loopCHECKFACET, 1, 1) - ((meshXYZ.mutable_at(loopCHECKFACET, 1, 1) - meshXYZ.mutable_at(loopCHECKFACET, 1, 2)) * (meshXYZ.mutable_at(loopCHECKFACET, 0, 1) - meshXYZ.mutable_at(loopCHECKFACET, 0, 0)) / (meshXYZ.mutable_at(loopCHECKFACET, 0, 1) - meshXYZ.mutable_at(loopCHECKFACET, 0, 2)));
					YRpredicted = meshXYZ.mutable_at(loopCHECKFACET, 1, 1) - ((meshXYZ.mutable_at(loopCHECKFACET, 1, 1) - meshXYZ.mutable_at(loopCHECKFACET, 1, 2)) * (meshXYZ.mutable_at(loopCHECKFACET, 0, 1) - gridCOx(loopX)) / (meshXYZ.mutable_at(loopCHECKFACET, 0, 1) - meshXYZ.mutable_at(loopCHECKFACET, 0, 2)));

					if ((Y1predicted > meshXYZ.mutable_at(loopCHECKFACET, 1, 0) && YRpredicted > gridCOy(loopY)) || (Y1predicted < meshXYZ.mutable_at(loopCHECKFACET, 1, 0) && YRpredicted < gridCOy(loopY))) {
						//The ray is on the same side of the 2 - 3 edge as the 1st vertex.
						Y2predicted = meshXYZ.mutable_at(loopCHECKFACET, 1, 2) - ((meshXYZ.mutable_at(loopCHECKFACET, 1, 2) - meshXYZ.mutable_at(loopCHECKFACET, 1, 0)) * (meshXYZ.mutable_at(loopCHECKFACET, 0, 2) - meshXYZ.mutable_at(loopCHECKFACET, 0, 1)) / (meshXYZ.mutable_at(loopCHECKFACET, 0, 2) - meshXYZ.mutable_at(loopCHECKFACET, 0, 0)));
						YRpredicted = meshXYZ.mutable_at(loopCHECKFACET, 1, 2) - ((meshXYZ.mutable_at(loopCHECKFACET, 1, 2) - meshXYZ.mutable_at(loopCHECKFACET, 1, 0)) * (meshXYZ.mutable_at(loopCHECKFACET, 0, 2) - gridCOx(loopX)) / (meshXYZ.mutable_at(loopCHECKFACET, 0, 2) - meshXYZ.mutable_at(loopCHECKFACET, 0, 0)));

						if ((Y2predicted > meshXYZ.mutable_at(loopCHECKFACET, 1, 1) && YRpredicted > gridCOy(loopY)) || (Y2predicted < meshXYZ.mutable_at(loopCHECKFACET, 1, 1) && YRpredicted < gridCOy(loopY))) {
							//The ray is on the same side of the 3 - 1 edge as the 2nd vertex.
							Y3predicted = meshXYZ.mutable_at(loopCHECKFACET, 1, 0) - ((meshXYZ.mutable_at(loopCHECKFACET, 1, 0) - meshXYZ.mutable_at(loopCHECKFACET, 1, 1)) * (meshXYZ.mutable_at(loopCHECKFACET, 0, 0) - meshXYZ.mutable_at(loopCHECKFACET, 0, 2)) / (meshXYZ.mutable_at(loopCHECKFACET, 0, 0) - meshXYZ.mutable_at(loopCHECKFACET, 0, 1)));
							YRpredicted = meshXYZ.mutable_at(loopCHECKFACET, 1, 0) - ((meshXYZ.mutable_at(loopCHECKFACET, 1, 0) - meshXYZ.mutable_at(loopCHECKFACET, 1, 1)) * (meshXYZ.mutable_at(loopCHECKFACET, 0, 0) - gridCOx(loopX)) / (meshXYZ.mutable_at(loopCHECKFACET, 0, 0) - meshXYZ.mutable_at(loopCHECKFACET, 0, 1)));

							if ((Y3predicted > meshXYZ.mutable_at(loopCHECKFACET, 1, 2) && YRpredicted > gridCOy(loopY)) || (Y3predicted < meshXYZ.mutable_at(loopCHECKFACET, 1, 2) && YRpredicted < gridCOy(loopY)))
								facetCROSSLIST.push_back(loopCHECKFACET);

						}
					}
				}

				// - 3 - Find the z coordinate of the locations where the ray crosses each facet or vertex:
				RowArrayX<double> gridCOzCROSS(facetCROSSLIST.size());
				std::vector<double> insideL;
				for (int index = 0; index < facetCROSSLIST.size(); index++) {
					// Define the equation describing the plane of the facet
					//    Ax + By + Cz + D = 0
					//    where  A = y1(z2 - z3) + y2(z3 - z1) + y3(z1 - z2)
					//           B = z1(x2 - x3) + z2(x3 - x1) + z3(x1 - x2)
					//           C = x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)
					//           D = -x1(y2 z3 - y3 z2) - x2(y3 z1 - y1 z3) - x3(y1 z2 - y2 z1)
					// For the x and y coordinates of the ray, solve these equations to find the z coordinate in this plane.
					int loopFINDZ = facetCROSSLIST.at(index);
					planecoA = meshXYZ.mutable_at(loopFINDZ, 1, 0) * (meshXYZ.mutable_at(loopFINDZ, 2, 1) - meshXYZ.mutable_at(loopFINDZ, 2, 2)) + meshXYZ.mutable_at(loopFINDZ, 1, 1) * (meshXYZ.mutable_at(loopFINDZ, 2, 2) - meshXYZ.mutable_at(loopFINDZ, 2, 0)) + meshXYZ.mutable_at(loopFINDZ, 1, 2) * (meshXYZ.mutable_at(loopFINDZ, 2, 0) - meshXYZ.mutable_at(loopFINDZ, 2, 1));
					planecoB = meshXYZ.mutable_at(loopFINDZ, 2, 0) * (meshXYZ.mutable_at(loopFINDZ, 0, 1) - meshXYZ.mutable_at(loopFINDZ, 0, 2)) + meshXYZ.mutable_at(loopFINDZ, 2, 1) * (meshXYZ.mutable_at(loopFINDZ, 0, 2) - meshXYZ.mutable_at(loopFINDZ, 0, 0)) + meshXYZ.mutable_at(loopFINDZ, 2, 2) * (meshXYZ.mutable_at(loopFINDZ, 0, 0) - meshXYZ.mutable_at(loopFINDZ, 0, 1));
					planecoC = meshXYZ.mutable_at(loopFINDZ, 0, 0) * (meshXYZ.mutable_at(loopFINDZ, 1, 1) - meshXYZ.mutable_at(loopFINDZ, 1, 2)) + meshXYZ.mutable_at(loopFINDZ, 0, 1) * (meshXYZ.mutable_at(loopFINDZ, 1, 2) - meshXYZ.mutable_at(loopFINDZ, 1, 0)) + meshXYZ.mutable_at(loopFINDZ, 0, 2) * (meshXYZ.mutable_at(loopFINDZ, 1, 0) - meshXYZ.mutable_at(loopFINDZ, 1, 1));
					planecoD = -meshXYZ.mutable_at(loopFINDZ, 0, 0) * (meshXYZ.mutable_at(loopFINDZ, 1, 1) * meshXYZ.mutable_at(loopFINDZ, 2, 2) - meshXYZ.mutable_at(loopFINDZ, 1, 2) * meshXYZ.mutable_at(loopFINDZ, 2, 1)) - meshXYZ.mutable_at(loopFINDZ, 0, 1) * (meshXYZ.mutable_at(loopFINDZ, 1, 2) * meshXYZ.mutable_at(loopFINDZ, 2, 0) - meshXYZ.mutable_at(loopFINDZ, 1, 0) \
						* meshXYZ.mutable_at(loopFINDZ, 2, 2)) - meshXYZ.mutable_at(loopFINDZ, 0, 2) * (meshXYZ.mutable_at(loopFINDZ, 1, 0) * meshXYZ.mutable_at(loopFINDZ, 2, 1) - meshXYZ.mutable_at(loopFINDZ, 1, 1) * meshXYZ.mutable_at(loopFINDZ, 2, 0));

					if (abs(planecoC) < 1e-14)
						planecoC = 0.0;
					gridCOzCROSS(index) = (-planecoD - planecoA * gridCOx(loopX) - planecoB * gridCOy(loopY)) / planecoC;
					if ((gridCOzCROSS(index) >= meshZmin - 1e-12) && (gridCOzCROSS(index) <= meshZmax + 1e-12))
						insideL.push_back(gridCOzCROSS(index));
				}
				//Remove values of gridCOzCROSS which are outside of the mesh limits(including a 1e-12 margin for error).
				std::sort(insideL.begin(), insideL.end());
				gridCOzCROSS = Eigen::Map<RowArrayX<double>>(insideL.data(), insideL.size());
				//Round gridCOzCROSS to remove any rounding errors, and take only the unique values :
				gridCOzCROSS = (gridCOzCROSS * 1e12).round() / 1e12;
				int aa = std::unique(gridCOzCROSS.data(), gridCOzCROSS.data() + gridCOzCROSS.size()) - gridCOzCROSS.data();
				gridCOzCROSS = Map<RowArrayX<double>>(gridCOzCROSS.data(), aa);

				// - 4 - Label as being inside the mesh all the voxels that the ray passes through after crossing one facet before crossing another facet
				// Only rays which cross an even number of facets are voxelised

				if ((gridCOzCROSS.size() % 2) == 0) {
					for (int loopASSIGN = 1; loopASSIGN < (gridCOzCROSS.size() / 2.0) + 1; loopASSIGN++) {
						for (int sz = 0; sz < gridCOz.size(); sz++) {
							if ((gridCOz(sz) > gridCOzCROSS(2 * loopASSIGN - 2)) && (gridCOz(sz) < gridCOzCROSS(2 * loopASSIGN - 1)))
								gridOUTPUT.mutable_at(loopX, loopY, sz) = 1;
						}
					}
				}

				// Remaining rays which meet the mesh in some way are not voxelised, but are labelled for correction later.
				else if (gridCOzCROSS.size() > 0) {
					correctionLIST.push_back(loopX);
					correctionLIST.push_back(loopY);
				}

			}

		}


	}
	//For rays where the voxelisation did not give a clear result, the ray is computed by interpolating from the surrounding rays.
	ssize_t countCORRECTIONLIST = correctionLIST.size() / 2;
	if (countCORRECTIONLIST > 0) {
		Map<Array<int, Dynamic, 2, RowMajor>> correctionArray(correctionLIST.data(), countCORRECTIONLIST, 2);
		//If necessary, add a one - pixel border around the x and y edges of thearray.
		//This prevents an error if the code tries to interpolate a ray at the edge of the x, y grid.
		if (correctionArray.colwise().minCoeff()(0) == 1 || correctionArray.colwise().maxCoeff()(0) == gridCOx.size() ||
			correctionArray.colwise().minCoeff()(1) == 1 || correctionArray.colwise().maxCoeff()(1) == gridCOy.size()) {
			// padding in X,Y. not in Z
			py::array_t<int> gridpadding = py::array_t<int>((voxcountX + 2) * (voxcountY + 2) * voxcountZ).attr("reshape")(std::make_tuple(voxcountX + 2, voxcountY + 2, voxcountZ));
			setConstValue(gridpadding, (int)0);
			gridpadding[py::make_tuple(py::slice(1, voxcountX + 1, 1),
				py::slice(1, voxcountY + 1, 1),
				py::slice(0, voxcountZ, 1))] = gridOUTPUT;
			gridOUTPUT = gridpadding;
			correctionArray = correctionArray + 1;

		}
		for (int loopC = 0; loopC < countCORRECTIONLIST; loopC++) {
			int lx = correctionArray(loopC, 0);
			int ly = correctionArray(loopC, 1);
			// Surrounding elements
			// t1 t2 t3
			// t4    t5
			// t6 t7 t8
			RowArrayX<int> t1 = toRowArray(gridOUTPUT[py::make_tuple(lx - 1, ly - 1, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t2 = toRowArray(gridOUTPUT[py::make_tuple(lx - 1, ly, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t3 = toRowArray(gridOUTPUT[py::make_tuple(lx - 1, ly + 1, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t4 = toRowArray(gridOUTPUT[py::make_tuple(lx, ly - 1, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t5 = toRowArray(gridOUTPUT[py::make_tuple(lx, ly + 1, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t6 = toRowArray(gridOUTPUT[py::make_tuple(lx + 1, ly - 1, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t7 = toRowArray(gridOUTPUT[py::make_tuple(lx + 1, ly, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			RowArrayX<int> t8 = toRowArray(gridOUTPUT[py::make_tuple(lx + 1, ly + 1, py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>());
			Array<int, 8, Dynamic, RowMajor> ttotal(8, t1.size());
			ttotal << t1, t2, t3, t4, t5, t6, t7, t8;

			RowArrayX<int> voxelsforcorrection = ttotal.colwise().sum();
			for (int ii = 0; ii < voxelsforcorrection.size(); ii++)
				if (voxelsforcorrection(ii) >= 4)
					gridOUTPUT.mutable_at(lx, ly, ii) = 1;
		}
		//Remove the one - pixel border surrounding the array, if this was added previously.
		if (gridOUTPUT.shape(0) > gridCOx.size() || gridOUTPUT.shape(1) > gridCOy.size())
			gridOUTPUT = gridOUTPUT[py::make_tuple(py::slice(1, voxcountX + 1, 1), py::slice(1, voxcountY + 1, 1), py::slice(0, voxcountZ, 1))].cast<py::array_t<int>>();
	}
	return gridOUTPUT;
}

bool Voxel::gen_vox_info() {
//	// Nodes sequence in one element
//	//    5 __________ 8
//	//    /          /|       z
//	//   /_________ / |       |
//	// 6|          |7 |       |____ y
//	//  |          |  |      /
//	//  |  1       |  |4    /
//	//  |          | /    x
//	//  |__________|/
//	// 2           3
#ifdef PYPRINT
	py::print("gen_vox_info start");
#endif // PYPRINT

	if (voxelgrid.size() < 1) {
		py::print("voxelgrid is not generated, run Voxel.VOXELISE first !!!");
		return 0;
	}
	py::array_t<int> ele_vox = voxelgrid.attr("transpose")(py::make_tuple(2, 1, 0));
	size_t NNX = ele_vox.shape(0);
	size_t NNY = ele_vox.shape(1);
	size_t NNZ = ele_vox.shape(2);
	nele = ele_vox.attr("sum")().cast<ssize_t>();
	py::array_t<int> nod_vox = py::array_t<int>((1 + NNX) * (1 + NNY) * (1 + NNZ)).attr("reshape")(std::make_tuple(1 + NNX, 1 + NNY, 1 + NNZ));
	py::array_t<int> nod_order = py::array_t<int>((1 + NNX) * (1 + NNY) * (1 + NNZ)).attr("reshape")(std::make_tuple(1 + NNX, 1 + NNY, 1 + NNZ));
	py::array_t<int> ele_nod_global = py::array_t<int>(NNX * NNY * NNZ * 8).attr("reshape")(std::make_tuple(NNX * NNY * NNZ, 8));
	setConstValue(nod_vox, 0);
	setConstValue(nod_order, 0);
	setConstValue(ele_nod_global, 0);
	
	Array<int, 8, 3, RowMajor> cube,cube_g;
	cube << 0, 0, 0,
		1, 0, 0,
		1, 1, 0,
		0, 1, 0,
		0, 0, 1,
		1, 0, 1,
		1, 1, 1,
		0, 1, 1;
	cube_g << 1, 1, 1,
		1, 0, 1,
		1, 0, 0,
		1, 1, 0,
		0, 1, 1,
		0, 0, 1,
		0, 0, 0,
		0, 1, 0;
	for (int i = 0; i < 8; i++)
	{
		nod_vox[py::make_tuple(py::slice(cube(i, 0), cube(i, 0) + NNX, 1), py::slice(cube(i, 1), cube(i, 1) + NNY, 1), py::slice(cube(i, 2), cube(i, 2) + NNZ, 1))]
			= nod_vox[py::make_tuple(py::slice(cube(i, 0), cube(i, 0) + NNX, 1), py::slice(cube(i, 1), cube(i, 1) + NNY, 1), py::slice(cube(i, 2), cube(i, 2) + NNZ, 1))].cast<py::array_t<int>>() + ele_vox;
	}

	// Generate the node matrix with global order
	int* ptr1 = (int*)nod_vox.request().ptr;
	int* ptr2 = (int*)nod_order.request().ptr;
	int order = 0;
	for (int i = 0; i < nod_vox.size(); i++)
		if (ptr1[i] > 0) { 
			ptr1[i] = 1; 
			ptr2[i] = order; 
			order++;
		}
	nnod = nod_vox.attr("sum")().cast<ssize_t>();

	// Loop all elementsand record the nodes order
	for (int i = 0; i < 8; i++)
	{
		ele_nod_global[py::make_tuple(py::slice(0, NNX*NNY*NNZ, 1), i)]
			= nod_order[py::make_tuple(py::slice(cube_g(i, 0), cube_g(i, 0) + NNX, 1), py::slice(cube_g(i, 1), cube_g(i, 1) + NNY, 1), py::slice(cube_g(i, 2), cube_g(i, 2) + NNZ, 1))].attr("flatten")();
	}
#ifdef PYPRINT
	py::print("nele", nele, "nnod", nnod);
#endif // PYPRINT
	ele_nod = py::array_t<int>(nele * 8).attr("reshape")(std::make_tuple(nele, 8));
	//ele_nod = ele_nod_global[ele_vox.attr("flatten")(), 0].cast<py::array_t<int>>();
	ele_vox = ele_vox.attr("flatten")().cast<py::array_t<int>>();
	order = 0;
	for (int i = 0; i < ele_vox.size(); i++)
		if (ele_vox.at(i) > 0) {
			ele_nod.mutable_at(order, 0) = ele_nod_global.mutable_at(i, 0);
			ele_nod.mutable_at(order, 1) = ele_nod_global.mutable_at(i, 1);
			ele_nod.mutable_at(order, 2) = ele_nod_global.mutable_at(i, 2);
			ele_nod.mutable_at(order, 3) = ele_nod_global.mutable_at(i, 3);
			ele_nod.mutable_at(order, 4) = ele_nod_global.mutable_at(i, 4);
			ele_nod.mutable_at(order, 5) = ele_nod_global.mutable_at(i, 5);
			ele_nod.mutable_at(order, 6) = ele_nod_global.mutable_at(i, 6);
			ele_nod.mutable_at(order, 7) = ele_nod_global.mutable_at(i, 7);
			order++;
		}

	nod_coor = py::array_t<int>(nnod * 3).attr("reshape")(std::make_tuple(nnod, 3));
	nod_coor_abs = py::array_t<double>(nnod * 3).attr("reshape")(std::make_tuple(nnod, 3));
	setConstValue(nod_coor, 0);
	setConstValue(nod_coor_abs, 0.0);
	order = 0;
	for (int k=0; k < NNX+1; k++)
		for (int j = 0; j < NNY + 1; j++)
			for (int i = 0; i < NNZ + 1; i++)
				if (nod_vox.mutable_at(k, j, i) > 0) {
					nod_coor.mutable_at(order, 0) = i;
					nod_coor.mutable_at(order, 1) = j;
					nod_coor.mutable_at(order, 2) = k;
					order++;
				}

	double dx = (__meshXmax - __meshXmin) / (NX + 0.5);
	double dy = (__meshYmax - __meshYmin) / (NY + 0.5);
	double dz = (__meshZmax - __meshZmin) / (NZ + 0.5);
	
	RowVectorXd lx = RowVectorXd::LinSpaced(NX + 1, __meshXmin, __meshXmin + dx * NX);
	RowVectorXd ly = RowVectorXd::LinSpaced(NY + 1, __meshYmin, __meshYmin + dy * NY);
	RowVectorXd lz = RowVectorXd::LinSpaced(NZ + 1, __meshZmin, __meshZmin + dz * NZ);

#pragma omp parallel for
	for (int i = 0; i < nnod; i++) {
		nod_coor_abs.mutable_at(i, 0) = lx(nod_coor.mutable_at(i, 0));
		nod_coor_abs.mutable_at(i, 1) = ly(nod_coor.mutable_at(i, 1));
		nod_coor_abs.mutable_at(i, 2) = lz(nod_coor.mutable_at(i, 2));
	}

	//nod_coor_abs[py::slice(0, nnod, 1), 0] = lx[nod_coor[py::slice(0, nnod, 1), 0]];
#ifdef PYPRINT
	py::print("gen_vox_info finished");
#endif // PYPRINT
	return 1;
}


void Voxel::save_mesh(std::string FILENAME)
{
	size_t pos = FILENAME.find_last_of(".");
	std::string exp = FILENAME.substr(pos);
	if(exp == ".inp")
		write_inp(FILENAME, 12, nod_coor_abs, ele_nod);
	else if (exp == ".vtk")
		write_vtk(FILENAME, 12, nod_coor_abs, ele_nod);
	else if (exp == ".vtu")
		write_vtu(FILENAME, 12, nod_coor_abs, ele_nod);
}

