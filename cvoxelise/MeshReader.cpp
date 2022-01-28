#include "MeshReader.h"


std::string& trim(std::string& str)
{
    if (str.empty())
        return str;

    str.erase(0, str.find_first_not_of(" \t\f\v\n\r"));
    str.erase(str.find_last_not_of(" \t\f\v\n\r") + 1);
    return str;
}

py::list Read_stl(std::string filename)
{
    stlInfo myinfo;
    py::list result;
    myinfo.stlFILENAME = filename;
    myinfo.stlFORMAT = stlGetFormat(myinfo.stlFILENAME);
    if (myinfo.stlFORMAT == "ascii")
        READ_stlascii(&myinfo);
    else
        READ_stlbinary(&myinfo);

    result.append(myinfo.coordVERTEXS);
    result.append(myinfo.coordNORMALS);
    result.append(myinfo.stlNAME);
    return result;
}

std::string stlGetFormat(std::string filename)
{
    std::ifstream fid;
    std::string FORMAT, header, tail;
    char buffer[80];
    fid.open(filename, std::ios::in | std::ios::binary);
    fid.seekg(0, std::ios::end);
    size_t fidSIZE = fid.tellg(); // Check the size of the file
    if ((fidSIZE - 84) % 50 > 0)
        FORMAT = "ascii";
    else {
        fid.seekg(0, std::ios::beg);//go to the beginning of the file
        fid.read(buffer, 80);
        header = buffer;
        bool isSolid = header.find("solid")+1;
        fid.seekg(-80, std::ios::end);          // go to the end of the file minus 80 characters
        fid.read(buffer, 80);
        tail = buffer;
        bool isEndSolid = tail.find("endsolid")+1;
        if (isSolid && isEndSolid) 
            FORMAT = "ascii";
        else FORMAT = "binary";
    }
    fid.close();
    return FORMAT;
}


void READ_stlascii(stlInfo* info)
{
    std::ifstream fid;
    std::string line, trimline;
    size_t facetcount = 0;
    std::vector<std::string> strNORMALS, strVERTEXS;
    py::array_t<float> coordVERTEXSall;
    auto local = py::dict();

    fid.open(info->stlFILENAME, std::ios::in);
    // Read the STL name
    std::getline(fid, line);
    trimline = trim(line);
    if (trimline.size() >= 7)
        info->stlNAME = trimline.substr(6);
    else info->stlNAME = "unnamed_object";

    while (std::getline(fid, line))
    {
        trimline = trim(line);
        if (trimline.find("facet normal") + 1)
        {
            facetcount += 1;
            strNORMALS.push_back(trimline);
        }
        if (trimline.find("vertex") + 1)
            strVERTEXS.push_back(trimline);
    }

    fid.close();

    info->coordNORMALS = py::array_t<double>(facetcount * 3).attr("reshape")(std::make_tuple(facetcount, 3));
    info->coordVERTEXS = py::array_t<double>(facetcount * 9).attr("reshape")(std::make_tuple(facetcount, 3, 3));
    float tempIN[12];
    std::stringstream ss;
    for (size_t loopF = 0; loopF < facetcount; loopF++)
    {
        ss  << strNORMALS.at(loopF).substr(12)
            << strVERTEXS.at(loopF*3).substr(6)
            << strVERTEXS.at(loopF*3 + 1).substr(6)
            << strVERTEXS.at(loopF*3 + 2).substr(6);
        ss >> tempIN[0] >> tempIN[1] >> tempIN[2]
            >> tempIN[3] >> tempIN[4] >> tempIN[5]
            >> tempIN[6] >> tempIN[7] >> tempIN[8]
            >> tempIN[9] >> tempIN[10] >> tempIN[11];
        ss.clear();

        // x, y, z components of the facet's normal vector
        info->coordNORMALS.mutable_data(loopF, 0)[0] = tempIN[0];
        info->coordNORMALS.mutable_data(loopF, 1)[0] = tempIN[1];
        info->coordNORMALS.mutable_data(loopF, 2)[0] = tempIN[2];
        // x, y, z coordinates of vertex 1
        info->coordVERTEXS.mutable_data(loopF, 0, 0)[0] = tempIN[3];
        info->coordVERTEXS.mutable_data(loopF, 1, 0)[0] = tempIN[4];
        info->coordVERTEXS.mutable_data(loopF, 2, 0)[0] = tempIN[5];
        // x, y, z coordinates of vertex 2
        info->coordVERTEXS.mutable_data(loopF, 0, 1)[0] = tempIN[6];
        info->coordVERTEXS.mutable_data(loopF, 1, 1)[0] = tempIN[7];
        info->coordVERTEXS.mutable_data(loopF, 2, 1)[0] = tempIN[8];
        // x, y, z coordinates of vertex 3
        info->coordVERTEXS.mutable_data(loopF, 0, 2)[0] = tempIN[9];
        info->coordVERTEXS.mutable_data(loopF, 1, 2)[0] = tempIN[10];
        info->coordVERTEXS.mutable_data(loopF, 2, 2)[0] = tempIN[11];
    }
}

void READ_stlbinary(stlInfo *info)
{
    std::ifstream fid;
    std::string line;
    size_t facetcount = 0;

    fid.open(info->stlFILENAME, std::ios::in | std::ios::binary);
    fid.seekg(80, std::ios::beg);              //go to the last 4 bytes of the header
    fid.read(reinterpret_cast<char*>(&facetcount), sizeof(int)); // Read the number of facets
    // Initialise arrays into which the STL data will be loaded :
    info->coordNORMALS = py::array_t<double>(facetcount * 3).attr("reshape")(std::make_tuple(facetcount, 3));
    info->coordVERTEXS = py::array_t<double>(facetcount * 9).attr("reshape")(std::make_tuple(facetcount, 3, 3));
    // Read the data for each facet:
    float tempIN[12];
    char temp[2];
    for (size_t loopF = 0; loopF < facetcount; loopF++)
    {
        fid.read(reinterpret_cast<char*>(&tempIN), sizeof(float) *12);// Read the data of each facet: 48 bytes
        // x, y, z components of the facet's normal vector
        info->coordNORMALS.mutable_data(loopF, 0)[0] = tempIN[0];
        info->coordNORMALS.mutable_data(loopF, 1)[0] = tempIN[1];
        info->coordNORMALS.mutable_data(loopF, 2)[0] = tempIN[2];
        // x, y, z coordinates of vertex 1
        info->coordVERTEXS.mutable_data(loopF, 0, 0)[0] = tempIN[3];
        info->coordVERTEXS.mutable_data(loopF, 1, 0)[0] = tempIN[4];
        info->coordVERTEXS.mutable_data(loopF, 2, 0)[0] = tempIN[5];
        // x, y, z coordinates of vertex 2
        info->coordVERTEXS.mutable_data(loopF, 0, 1)[0] = tempIN[6];
        info->coordVERTEXS.mutable_data(loopF, 1, 1)[0] = tempIN[7];
        info->coordVERTEXS.mutable_data(loopF, 2, 1)[0] = tempIN[8];
        // x, y, z coordinates of vertex 3
        info->coordVERTEXS.mutable_data(loopF, 0, 2)[0] = tempIN[9];
        info->coordVERTEXS.mutable_data(loopF, 1, 2)[0] = tempIN[10];
        info->coordVERTEXS.mutable_data(loopF, 2, 2)[0] = tempIN[11];
        fid.read(temp,2);   // Move to the start of the next facet.
    }
    fid.close();
    info->stlNAME = "unnamed_object";
}


