#include "auxiliar.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace std;
using namespace Eigen;
namespace bf = boost::filesystem;

void readImagesFZU(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps);

int main(int argc, char const* argv[])
{
    if (argc < 3) {
        LOGE("Usage: " << argv[0] << " <image_folder> <odom_file> [output_odom_file]");
        return -1;
    }

    string imgFolder(argv[1]);
    string odomFile(argv[2]);
    LOGI("Reading odometries from file: " << odomFile << ", with image folder: " << imgFolder);

    string outputFile = (argc > 3) ? string(argv[3]) : (imgFolder + "/../odom_sync.txt");
    LOGI("Output synced odometries file is: " << outputFile);

    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    readImagesFZU(imgFolder, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();
    if (nImages < 1) {
        LOGE("Failed to load images!");
        return -1;
    }

    ifstream rec(odomFile);
    if (!rec.is_open()) {
        LOGE("Please check if the ODOM file exists!");
        return -1;
    }

    vector<Vector4d> vOdometries;
    vOdometries.reserve(20000);

    string line;
    double trans[3];
    double quat[4];
    double angle[3];
    double timestamp;
    while (std::getline(rec, line), !line.empty()) {
        stringstream iss(line);
        iss >> timestamp >> trans[0] >> trans[1] >> trans[2] >> quat[0] >> quat[1] >> quat[2] >> quat[3];  // [m]
        Eigen::Quaterniond tmp_quat(quat[3], quat[0], quat[1], quat[2]);
        Eigen::Vector3d tmp_angle = tmp_quat.matrix().eulerAngles(2, 1, 0);
        angle[0] = NormalizeAngle(tmp_angle[0]);
        angle[1] = NormalizeAngle(tmp_angle[1]);
        angle[2] = NormalizeAngle(tmp_angle[2]);

        vOdometries.emplace_back(timestamp, trans[0], trans[1], angle[0]);
        assert(abs(trans[2] - 0.) < 1e-6);
        line.clear();   //! NOTE clear line in case of infinite loop!
    }
    rec.close();
    LOGI("Read " << vOdometries.size() << " odometry data.");
    if (vOdometries.size() == 0) {
        return -1;
    }

    bool bSorted = is_sorted(vTimestamps.begin(), vTimestamps.end());  // ASC (ascending order default)
    if (!bSorted) {
        LOGW("Image name is not sorted! They will be sorted right now!");
        sort(vTimestamps.begin(), vTimestamps.end());
    } else {
        LOGI("Image name is already sorted!");
    }

    // sync data with upper_bound
    vector<Vector4d> vOdometryFiltered;
    vOdometryFiltered.reserve(vTimestamps.size());

    int skipFrames = 0;
    int j = 0;
    for (size_t i = 0; i < vTimestamps.size(); ++i) {
        const double t0 = vTimestamps[i];
        const size_t r = distance(vOdometries.begin(), find_if(vOdometries.begin(), vOdometries.end(),
                                                               [t0](const Vector4d& value) { return value[0] >= t0; }));
        if (r > vOdometries.size() - 1) {
            LOGW("Skip frame #" << i << ", cause no odometry associated with it.");
            j > 0 ? vOdometryFiltered.push_back(vOdometryFiltered[vOdometryFiltered.size()-1]) : vOdometryFiltered.emplace_back(0, 0, 0, 0);
            continue;
        }

        if (r == 0) {
            vOdometryFiltered.push_back(vOdometries[0]);
            continue;
        }

        // linear interpolation
        const size_t l = r - 1;
        double alpha = (t0 - vOdometries[l][0]) / (vOdometries[r][0] - vOdometries[l][0]);
        double tx = vOdometries[l][1] + alpha * (vOdometries[r][1] - vOdometries[l][1]);
        double ty = vOdometries[l][2] + alpha * (vOdometries[r][2] - vOdometries[l][2]);
        double ry = vOdometries[l][3] + alpha * (vOdometries[r][3] - vOdometries[l][3]);
        vOdometryFiltered.emplace_back(t0, tx, ty, ry);
    }

    // write to outputFile
    ofstream ofs(outputFile, ios_base::out);
    if (!ofs.is_open()) {
        LOGE("ERROR on opening output file! Please check!");
        return -1;
    }

    ofs << setiosflags(ios::fixed) << setprecision(9);
    for (size_t j = 0; j < vOdometryFiltered.size(); ++j) {
        Vector4d& data = vOdometryFiltered[j];
        ofs << data[0] << "  " << data[1] << "  " << data[2] << "  " << data[3] << endl;
    }
    ofs.close();

    LOGI("Done.");

    return 0;
}


void readImagesFZU(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps)
{
    bf::path path(strImagePath);
    if (!bf::exists(path)) {
        LOGE("Data folder doesn't exist!" << strImagePath);
        return;
    }

    vector<pair<string, double>> vstrImgTime;
    vstrImgTime.reserve(2000);

    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;
        if (bf::is_regular_file(iter->status())) {
            // format: /s.ns.jpg
            string s = iter->path().string();
            size_t i = s.find_last_of('/');
            size_t j = s.find_last_of('.');
            if (i == string::npos || j == string::npos)
                continue;
            string strTimeStamp = s.substr(i + 1, j);
            double t = atof(strTimeStamp.c_str());
            vstrImgTime.emplace_back(s, t);  // us
        }
    }

    sort(vstrImgTime.begin(), vstrImgTime.end(),
         [&](const pair<string, double>& lf, const pair<string, double>& rf) { return lf.second < rf.second; });

    const size_t numImgs = vstrImgTime.size();
    vTimeStamps.resize(numImgs);
    vstrImages.resize(numImgs);
    for (size_t k = 0; k < numImgs; ++k) {
        vstrImages[k] = vstrImgTime[k].first;
        vTimeStamps[k] = vstrImgTime[k].second;
    }

    if (vstrImages.empty()) {
        LOGE("Not image data in the folder!");
        return;
    } else {
        LOGI("Read " << vstrImages.size() << " image files in the folder");
    }
}