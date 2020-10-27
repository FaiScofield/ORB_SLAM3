#include "Frame.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "CameraModels/Pinhole.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#define Adaptive_histogram_Equalization 1

using namespace ORB_SLAM3;
using namespace cv;
using namespace std;
namespace bf = boost::filesystem;

string g_vocabularyFile = "/home/vance/slam_ws/ORB_SLAM3/Vocabulary/ORBvoc.bin";

void readImagesRK(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps);

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
                             "{folder    f| |data folder}"
                             "{help      h|false|show help message}");

    if (argc < 2 || parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    const String strFolder = parser.get<String>("folder");
    cout << " - data folder: " << strFolder << endl;

    // vector<string> vStrImages;
    // vector<double> vTimeStamps;
    // readImagesRK(strFolder, vStrImages, vTimeStamps);
    vector<String> vStrImages;
    cv::glob(strFolder, vStrImages);

#if Adaptive_histogram_Equalization
    Ptr<CLAHE> claher = createCLAHE(2.0, Size(6, 6));
#endif
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = 3.77044e-02;
    DistCoef.at<float>(1) = -3.261434e-02;
    DistCoef.at<float>(2) = -9.238135e-04;
    DistCoef.at<float>(3) = 5.452823e-04;
    vector<float> vCamCalib{207.9359613169054, 207.4159055585876, 160.5827136112504, 117.7328673795551};
    GeometricCamera* pCamera = new Pinhole(vCamCalib);
    ORBextractor* pDetector = new ORBextractor(500, 1.5, 3, 17, 12);
    ORBmatcher* pMatcher = new ORBmatcher(0.7, true);
    ORBVocabulary* pVocabulary = new ORBVocabulary();
    if (!pVocabulary->loadFromBinaryFile(g_vocabularyFile)) {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << g_vocabularyFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    Frame frameCur, frameRef;
    Mat img, gray, outImg;
    vector<Point2f> vPrevMatched;
    vector<int> vMatches12;
    vector<DMatch> vMatches;
    vector<int> vLappingArea = {0, 1000};
    for (int k = 0, kend = vStrImages.size(); k < kend; ++k) {
        cout << "# " << k << " Dealing with img " << vStrImages[k] << endl;

        img = imread(vStrImages[k], IMREAD_COLOR);
        cvtColor(img, gray, COLOR_BGR2GRAY);
#if Adaptive_histogram_Equalization
        claher->apply(gray, gray);
#endif

        frameCur = Frame(gray, 0, pDetector, pVocabulary, pCamera, DistCoef, 0, 0);
        frameCur.imgLeft = gray.clone();
        if (k > 0) {
            pMatcher->SearchForInitialization(frameRef, frameCur, vPrevMatched, vMatches12, 50);

            int nMatches = 0;
            vMatches.clear();
            vMatches.reserve(vMatches12.size());
            for (size_t i = 0; i < vMatches12.size(); ++i) {
                if (vMatches12[i] >= 0 && vMatches12[i] < (int)frameCur.mvKeys.size()) {
                    vMatches.emplace_back(i, vMatches12[i], 1);
                    nMatches++;
                }
            }
            cout << "# " << k << " matches : " << nMatches << endl;

            hconcat(frameRef.imgLeft, frameCur.imgLeft, outImg);
            drawMatches(frameRef.imgLeft, frameRef.mvKeys, frameCur.imgLeft, frameCur.mvKeys, vMatches, outImg);
            char waterMark[64];
            snprintf(waterMark, 64, "Idx: %d - %d, matches: %d", k/10, k, nMatches);
            putText(outImg, waterMark, Point(20,20), 0, 1, Scalar(0,0,255));
            imshow("ORB features matches", outImg);
            waitKey(100);
        }

        // swap keyframe
        KeyPoint::convert(frameCur.mvKeys, vPrevMatched);
        if (k % 10 == 0) {
            frameRef = frameCur;
            vMatches12.swap(vMatches12);
        }
    }

    cout << "Done." << endl;
    return 0;
}


void readImagesRK(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps)
{
    bf::path path(strImagePath);
    if (!bf::exists(path)) {
        cerr << "[Main ][Error] Data folder doesn't exist!" << endl;
        return;
    }

    vector<pair<string, long long>> vstrImgTime;
    vstrImgTime.reserve(3000);

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

    sort(vstrImgTime.begin(), vstrImgTime.end(),
         [&](const pair<string, long long>& lf, const pair<string, long long>& rf) {
             return lf.second < rf.second;
         });

    const size_t numImgs = vstrImgTime.size();
    if (!numImgs) {
        cerr << "[Main ][Error] Not image data in the folder!" << endl;
        return;
    } else {
        cout << "[Main ][Info ] Read " << numImgs << " image files in the folder." << endl;
    }

    vTimeStamps.resize(numImgs);
    vstrImages.resize(numImgs);
    for (size_t k = 0; k < numImgs; ++k) {
        vstrImages[k] = vstrImgTime[k].first;
        vTimeStamps[k] = (double)vstrImgTime[k].second / 1e3;
    }
}
