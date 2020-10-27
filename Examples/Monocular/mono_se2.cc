/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel
 * and Juan D. Tardós, University of Zaragoza. Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M.
 * Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#include <algorithm>
#include <boost/filesystem.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

#include <opencv2/core/core.hpp>

#include <System.h>

using namespace std;
namespace bf = boost::filesystem;

void LoadImages(const string& strImagePath, int imgNum, vector<string>& vstrImages, vector<double>& vTimeStamps);
void readImagesRK(const string& strImagePath, int imgNum, vector<string>& vstrImages, vector<double>& vTimeStamps);

int main(int argc, char** argv)
{
    if (argc < 4) {
        cerr << endl
             << "Usage: ./mono_se2 path_to_vocabulary path_to_settings path_to_sequence_folder_1 "
             // " (path_to_image_folder_2 path_to_times_file_2 ... "
             // "path_to_image_folder_N path_to_times_file_N) (trajectory_file_name)"
             << endl;
        return 1;
    }

    // const int num_seq = (argc - 3) / 2;
    // cout << "num_seq = " << num_seq << endl;
    // bool bFileName = (((argc - 3) % 2) == 1);
    // string file_name;
    // if (bFileName) {
    //     file_name = string(argv[argc - 1]);
    //     cout << "file name: " << file_name << endl;
    // }

    const int num_seq = 1;
    bool bFileName = false;

    // Load all sequences:
    int seq;
    vector<vector<string>> vstrImageFilenames;
    vector<vector<double>> vTimestampsCam;
    vector<int> nImages;

    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    nImages.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq < num_seq; seq++) {
        cout << "Loading images for sequence " << seq << "...";
        LoadImages(string(argv[3]) + "/image/", 3108, vstrImageFilenames[seq], vTimestampsCam[seq]);
        //        readImagesRK(string(argv[3]) + "/slamimg/", 1969, vstrImageFilenames[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);

    // camera matirx for se2 dataset
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.5, cv::Size(8, 8));
    cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);
    cv::Mat D = cv::Mat::zeros(4, 1, CV_32FC1);
    K.at<float>(0, 0) = 231.976033627644090;
    K.at<float>(1, 1) = 232.157224036901510;
    K.at<float>(0, 2) = 326.923920970539310;
    K.at<float>(1, 2) = 227.838488395348380;
    D.at<float>(0, 0) = -0.207406506100898;
    D.at<float>(1, 0) = 0.032194071349429;
    D.at<float>(2, 0) = 0.001120166051888;
    D.at<float>(3, 0) = 0.000859411522110;
    cout << endl << "K = " << K << endl;
    cout << "D = " << D << endl;
    for (seq = 0; seq < num_seq; seq++) {

        // Main loop
        cv::Mat im, imUn;
        int proccIm = 0;
        for (int ni = 0; ni < nImages[seq]; ni++, proccIm++) {

            // Read image from file
//            im = cv::imread(vstrImageFilenames[seq][ni], CV_LOAD_IMAGE_UNCHANGED);
            im = cv::imread(vstrImageFilenames[seq][ni], CV_LOAD_IMAGE_GRAYSCALE);
            double tframe = vTimestampsCam[seq][ni];

            if (im.empty()) {
                cerr << endl << "Failed to load image at: " << vstrImageFilenames[seq][ni] << endl;
                return 1;
            }

            // undistort image
            cv::undistort(im, imUn, K, D);
            clahe->apply(imUn, imUn);
//            cv::imshow("IMAGE UNDISTORT", imUn);
//            cv::waitKey(5);

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

            // Pass the image to the SLAM system
            // cout << "tframe = " << tframe << endl;
            SLAM.TrackMonocular(imUn, tframe);  // TODO change to monocular_inertial

#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

            double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

            vTimesTrack[ni] = ttrack;

            // Wait to load the next frame
            double T = 0;
            if (ni < nImages[seq] - 1)
                T = vTimestampsCam[seq][ni + 1] - tframe;
            else if (ni > 0)
                T = tframe - vTimestampsCam[seq][ni - 1];

            if (ttrack < T)
                usleep((T - ttrack) * 1e6);  // 1e6
        }

        if (seq < num_seq - 1) {
            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }
    }
    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    if (bFileName) {
        const string kf_file = "kf_" + string(argv[argc - 1]) + ".txt";
        const string f_file = "f_" + string(argv[argc - 1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    } else {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    return 0;
}

void LoadImages(const string& strImagePath, int imgNum, vector<string>& vstrImages, vector<double>& vTimeStamps)
{
    vTimeStamps.reserve(imgNum + 1);
    vstrImages.reserve(imgNum + 1);

    double timeStamp = 1403636579763555584 / 1e9;  // s
    for (int k = 0; k < imgNum; ++k) {
        vstrImages.push_back(strImagePath + to_string(k) + ".bmp");
        timeStamp += 0.030;  // 30ms per frame
        vTimeStamps.push_back(timeStamp);
    }
}

static inline bool lessThen(const pair<string, long long>& lf, const pair<string, long long>& rf)
{
    return lf.second < rf.second;
}

void readImagesRK(const string& strImagePath, int imgNum, vector<string>& vstrImages, vector<double>& vTimeStamps)
{
    bf::path path(strImagePath);
    if (!bf::exists(path)) {
        cerr << "[Main ][Error] Data folder doesn't exist!" << endl;
        return;
    }


    double timeStamp = 1403636579763555584 / 1e9;  // s
    for (int k = 0; k < imgNum; ++k) {
        timeStamp += 0.030;  // 30ms per frame
        vTimeStamps.push_back(timeStamp);
    }

    vector<pair<string, long long>> vstrImgTime;
    vstrImgTime.reserve(imgNum + 1);

    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;
        if (bf::is_regular_file(iter->status())) {
            // format: /frameRaw12987978101.jpg
            string s = iter->path().string();
            size_t i = s.find_last_of('w');
            size_t j = s.find_last_of('.');
            if (i == string::npos || j == string::npos)
                continue;
            auto t = atoll(s.substr(i + 1, j - i - 1).c_str());
            vstrImgTime.emplace_back(s, t);
        }
    }

    sort(vstrImgTime.begin(), vstrImgTime.end(), lessThen);

    const size_t numImgs = vstrImgTime.size();
    vTimeStamps.resize(numImgs);
    vstrImages.resize(numImgs);
    for (size_t k = 0; k < numImgs; ++k) {
        vstrImages[k] = vstrImgTime[k].first;
        vTimeStamps[k] = (double)vstrImgTime[k].second / 1e3;
    }

    if (vstrImages.empty()) {
        cerr << "[Main ][Error] Not image data in the folder!" << endl;
        return;
    } else {
        cout << "[Main ][Info ] Read " << vstrImages.size() << " image files in the folder." << endl;
    }
}
