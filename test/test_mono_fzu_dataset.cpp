#include "System.h"
// #include "Odometry.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

#define ENABLE_EQUALIZE_HISTOGRAM   1

using namespace std;
namespace bf = boost::filesystem;

void readImagesFZU(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps);

int main(int argc, char** argv)
{
    const string vocabularyFile = "/home/vance/slam_ws/PL-SLAM-Mono/Vocabulary/ORBvoc.bin";
    string settingFile;
    string sequenceFolder;
    string odomRawFile;

    int startIdx = 0;
    if (argc < 3) {
        cout << "use default parameters..." << endl;
        settingFile = "/home/vance/dataset/fzu/ORB-SLAM3-Config.yaml";
        sequenceFolder = "/home/vance/dataset/fzu/201224_hall_3/image/";
        odomRawFile = "/home/vance/dataset/fzu/201224_hall_3/odom_sync.txt";
    } else {
        settingFile     = string(argv[1]);
        sequenceFolder  = string(argv[2]);
        if (argc >= 4) {
            odomRawFile = string(argv[3]);
        }
        if (argc >= 5) {
            startIdx = atoi(argv[4]);
            startIdx = max(0, startIdx);
        }
    }
    cout << "sequence: " << sequenceFolder << endl;
    cout << "settingFile: " << settingFile << endl;
    cout << "odomRawFile: " << odomRawFile << endl;
    cout << "startIdx: " << startIdx << endl;

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    readImagesFZU(sequenceFolder, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();
    if (nImages < 1) {
        cerr << "ERROR: Failed to load images!" << endl;
        return -1;
    }

    bool bWithOdom = false;
    ifstream rec(odomRawFile);
    if (rec.is_open()) {
        bWithOdom = true;
    } else {
        cerr << "[Main ][Error] Please check if the file exists!" << odomRawFile << endl;
    }

#if 1
    vector<ORB_SLAM3::ODOM::Point> vOdometries;
    int nOdoms = 0;
    if (bWithOdom) {
        vOdometries.reserve(nImages);
        float x, y, theta;
        double timestamp;
        string line;
        while (std::getline(rec, line), !line.empty()) {
            istringstream iss(line);
            iss >> timestamp >> x >> y >> theta; // [m],[rad]
            vOdometries.emplace_back(x, y, theta, timestamp);
            line.clear();
        }
        nOdoms = static_cast<int>(vOdometries.size());
    
        if (nOdoms < 1 || nOdoms < nImages) {
            cerr << "ERROR: Failed to load odometries! nOdoms = " << nOdoms << endl;
            bWithOdom = false;
        }
    }
#endif

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    auto sensorType = bWithOdom ? ORB_SLAM3::System::ODOM_MONOCULAR : ORB_SLAM3::System::MONOCULAR;
    ORB_SLAM3::System SLAM(vocabularyFile, settingFile, sensorType, true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl;
    cout << "Odometries in the sequence: " << nOdoms << endl << endl;

    cv::Ptr<cv::CLAHE> pClaher = cv::createCLAHE(3, cv::Size(8, 8));


    // Main loop
    cv::Mat im, imCa;
    for (int ni = startIdx; ni < nImages; ni++) {
        cout << "========================= " << ni << " / " << nImages << " =========================" << endl;

        // Read image from file
        im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_GRAYSCALE);
        double tframe = vTimestamps[ni];    // [s]

        if (im.empty()) {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

#if ENABLE_EQUALIZE_HISTOGRAM
        // char fileName[128];
        // snprintf(fileName, 128, "/home/vance/output/img_%d_source.png", ni);
        // imwrite(fileName, im);
        pClaher->apply(im, imCa);
        im = imCa;
        // snprintf(fileName, 128, "/home/vance/output/img_%d_clahed.png", ni);
        // imwrite(fileName, im);
#endif

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        if (bWithOdom) {
            vector<ORB_SLAM3::ODOM::Point> vOdomsThisFrame;
            vOdomsThisFrame.push_back(vOdometries[ni]);
            SLAM.TrackMonocularWithOdom(im, tframe, vOdomsThisFrame);
        } else {
            SLAM.TrackMonocular(im, tframe);
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        vTimesTrack[ni] = ttrack;   // [s]

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);  // usleep([us])
        
        // cout << "============================================================" << endl;
    }
    cv::waitKey(0);

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++) {
        totaltime += vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void readImagesFZU(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps)
{
    bf::path path(strImagePath);
    if (!bf::exists(path)) {
        cerr << "[Main ][Error] Data folder doesn't exist!" << endl;
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
            vstrImgTime.emplace_back(s, t); // us
        }
    }

    sort(vstrImgTime.begin(), vstrImgTime.end(),
         [&](const pair<string, double>& lf, const pair<string, double>& rf) {
            return lf.second < rf.second;}
    );

    const size_t numImgs = vstrImgTime.size();
    vTimeStamps.resize(numImgs);
    vstrImages.resize(numImgs);
    for (size_t k = 0; k < numImgs; ++k) {
        vstrImages[k] = vstrImgTime[k].first;
        vTimeStamps[k] = vstrImgTime[k].second;
    }

    if (vstrImages.empty()) {
        cerr << "[Main ][Error] Not image data in the folder!" << endl;
        return;
    } else {
        cout << "[Main ][Info ] Read " << vstrImages.size() << " image files in the folder." << endl;
    }
}