#include "MeshWriter.h"

void write_inp(std::string FILENAME, int type, py::array_t<double> nodes, py::array_t<int> elements) {
	int ele_offset = 1;
	FILE* fp;
	fp = fopen(FILENAME.c_str(), "w+");
	fprintf(fp, "**\n*Node\n");
	for (int i = 0; i < nodes.shape(0); i++)
		fprintf(fp, "%d,   %.6f, %.6f, %.6f\n", i + 1,
			nodes.mutable_at(i, 0),
			nodes.mutable_at(i, 1),
			nodes.mutable_at(i, 2));
	fprintf(fp, "\n*Element, type=%s\n", VTK_TO_ABAQUS_TYPE.at(type).c_str());
	for (int j = 0; j < elements.shape(0); j++)
		fprintf(fp, "%d,   %d, %d, %d, %d, %d, %d, %d, %d\n", j + 1,
			elements.mutable_at(j, 0) + ele_offset,
			elements.mutable_at(j, 1) + ele_offset,
			elements.mutable_at(j, 2) + ele_offset,
			elements.mutable_at(j, 3) + ele_offset,
			elements.mutable_at(j, 4) + ele_offset,
			elements.mutable_at(j, 5) + ele_offset,
			elements.mutable_at(j, 6) + ele_offset,
			elements.mutable_at(j, 7) + ele_offset);
	fclose(fp);
}

void write_vtk(std::string FILENAME, int type, py::array_t<double> nodes, py::array_t<int> elements) {
	FILE* fp;
	fp = fopen(FILENAME.c_str(), "w+");
	fprintf(fp, 
		"# vtk DataFile Version 3.0\n"
		"Volume Mesh\n"
		"ASCII\n"
		"DATASET UNSTRUCTURED_GRID\n");

	fprintf(fp, "POINTS  %d  float\n", int(nodes.shape(0)) );
	for (int i = 0; i < nodes.shape(0); i++)
		fprintf(fp, "%.9f  %.9f  %.9f\n",
			nodes.mutable_at(i, 0),
			nodes.mutable_at(i, 1),
			nodes.mutable_at(i, 2));
	
	fprintf(fp, "CELLS  %d  %d\n", int(elements.shape(0)), int(elements.size() + elements.shape(0)));
	switch (elements.shape(1))
	{
	case 4:
		for (int j = 0; j < elements.shape(0); j++)
			fprintf(fp, "%d %d %d %d %d\n", 4,
				elements.mutable_at(j, 0) ,
				elements.mutable_at(j, 1) ,
				elements.mutable_at(j, 2) ,
				elements.mutable_at(j, 3) );
		break;
	case 8:
		for (int j = 0; j < elements.shape(0); j++)
			fprintf(fp, "%d %d %d %d %d %d %d %d %d\n", 8,
				elements.mutable_at(j, 0) ,
				elements.mutable_at(j, 1) ,
				elements.mutable_at(j, 2) ,
				elements.mutable_at(j, 3) ,
				elements.mutable_at(j, 4) ,
				elements.mutable_at(j, 5) ,
				elements.mutable_at(j, 6) ,
				elements.mutable_at(j, 7) );
		break;
	}

	fprintf(fp, "CELL_TYPES  %d\n", int(elements.shape(0)));
	for (int j = 0; j < elements.shape(0); j++)
		fprintf(fp, "%d\n", type);

	fclose(fp);
}

void write_vtu(std::string FILENAME, int type, py::array_t<double> nodes, py::array_t<int> elements) {
	FILE* fp;
	int header,hlen;
	fp = fopen(FILENAME.c_str(), "w+");
	fprintf(fp,
		"<?xml version = \"1.0\"?>\n"
		"<VTKFile type = \"UnstructuredGrid\" version = \"0.1\" byte_order = \"LittleEndian\">\n"
		"<UnstructuredGrid>\n"
		"<Piece NumberOfPoints = \"%d\" NumberOfCells = \"%d\"> \n", int(nodes.shape(0)), int(elements.shape(0)));

	fprintf(fp, 
		"<Points>\n"
		"<DataArray type = \"Float32\" Name = \"Points\" NumberOfComponents = \"3\" format = \"binary\">\n");
	
	py::array_t<float> newnodes = nodes.cast<py::array_t<float>>();

	char* encoding = base64((char*)newnodes.data(), sizeof(float) * nodes.size(), &header);
	fprintf(fp, base64((char*)&header, sizeof(int), &hlen));
	fprintf(fp, encoding);

	fprintf(fp, 
		"\n"
		"</DataArray>\n"
		"</Points>\n"
		"<Cells>\n"
		"<DataArray type = \"Int32\" Name = \"connectivity\" format = \"binary\">\n");

	encoding = base64((char*)elements.data(), sizeof(int) * elements.size(), &header);
	fprintf(fp, base64((char*)&header, sizeof(int), &hlen));
	fprintf(fp, encoding);

	py::array_t<int> offsets = py::array_t<int>(elements.shape(0));
	py::array_t<int> types = py::array_t<int>(elements.shape(0));

#pragma omp parallel for
	for (int i = 0; i < elements.shape(0); i++) {
		offsets.mutable_at(i) = int(elements.shape(1)) * (1 + i);
		types.mutable_at(i) = type;
	}
	
	fprintf(fp,
		"\n"
		"</DataArray>\n"
		"<DataArray type = \"Int32\" Name = \"offsets\" format = \"binary\">\n");

	encoding = base64((char*)offsets.data(), sizeof(int) * offsets.size(), &header);
	fprintf(fp, base64((char*)&header, sizeof(int), &hlen));
	fprintf(fp, encoding);

	fprintf(fp,
		"\n"
		"</DataArray>\n"
		"<DataArray type = \"Int32\" Name = \"types\" format = \"binary\">\n");

	encoding = base64((char*)types.data(), sizeof(int) * types.size(), &header);
	fprintf(fp, base64((char*)&header, sizeof(int), &hlen));
	fprintf(fp, encoding);

	fprintf(fp,
		"\n"
		"</DataArray>\n"
		"</Cells>\n"
		"</Piece>\n"
		"</UnstructuredGrid>\n"
		"</VTKFile>\n");

	fclose(fp);
	return;
}