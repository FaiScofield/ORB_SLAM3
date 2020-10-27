#include "ORBextractor.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#define DEFAULT_DATASET_FOLDER "/home/vance/dataset/se2/DatasetRoom/image/"

using namespace ORB_SLAM3;
using namespace cv;
using namespace std;
namespace bf = boost::filesystem;

void readImagesRK(const string& strImagePath, vector<string>& vstrImages,
                  vector<double>& vTimeStamps);
void readImagesSE2(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps);

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
                             "{folder    f| |data folder}"
                             "{undistort u|false|undistort image}"
                             "{equalize  e|false|equalize image histogram}"
                             "{help      h|false|show help message}");

    if (/* argc < 2 ||  */ parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    String strFolder = parser.get<String>("folder");
    if (strFolder.empty())
        strFolder = string(DEFAULT_DATASET_FOLDER);
    cout << " - data folder: " << strFolder << endl;

    bool bUndistortImg = parser.get<bool>("undistort");
    cout << " - bUndistortImg: " << bUndistortImg << endl;
    bool bEqualizeImg = parser.get<bool>("equalize");
    cout << " - bEqualizeImg: " << bEqualizeImg << endl;

    vector<string> vStrImages;
    vector<double> vTimeStamps;
    // readImagesRK(strFolder, vStrImages, vTimeStamps);
    // cv::glob(strFolder, vStrImages);
    // sort(vStrImages.begin(), vStrImages.end(),
    //         [&](const String& lhs, const String& rhs) { 
    //             const auto idx1 = lhs.find_last_of('.');
    //             const auto idx2 = rhs.find_last_of('.');
    //             return atoi(lhs.substr(0, idx1).c_str()) < atoi(rhs.substr(0, idx2).c_str()); 
    //         }
    // );
    readImagesSE2(strFolder, vStrImages, vTimeStamps);

    ORBextractor detector(500, 2, 3, 17, 12);
    Ptr<CLAHE> claher = createCLAHE(2.0, Size(6, 6));

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

    Mat img, gray, outImg;
    Mat descriptors;
    vector<KeyPoint> vKPs;
    vector<int> vLappingArea = {0, 0};
    for (int k = 0, kend = vStrImages.size(); k < kend; ++k) {
        cout << "# " << k << " Dealing with img " << vStrImages[k] << endl;

        img = imread(vStrImages[k], IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Open image error: " << vStrImages[k] << endl;
            continue;
        }
        cvtColor(img, gray, COLOR_BGR2GRAY);

        if (bUndistortImg) {
            Mat imgUn;
            cv::undistort(gray, imgUn, K, D);
            imgUn.copyTo(gray);
        }
        if (bEqualizeImg)
            claher->apply(gray, gray);

        detector(gray, cv::noArray(), vKPs, descriptors, vLappingArea);
        cout << "# " << k << " Number ORB features: " << vKPs.size() << endl;
        
        drawKeypoints(gray, vKPs, outImg, Scalar(0, 255, 0));
        imshow("ORB features distribution", outImg);
        waitKey(30);
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

void readImagesSE2(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps)
{
    bf::path path(strImagePath);
    if (!bf::exists(path)) {
        cerr << "[Main ][Error] Data folder doesn't exist!" << endl;
        return;
    }

    vstrImages.resize(3108);
    vTimeStamps.resize(3108);
    double time = 6000000 * 1e-9;
    for (int i = 0; i < 3108; ++i) {
        vstrImages[i] = strImagePath + "/" + to_string(i) + ".bmp";
        vTimeStamps[i] = time;
        time += 0.03 * 1e-9;
    }


    const size_t numImgs = vstrImages.size();
    if (!numImgs) {
        cerr << "[Main ][Error] Not image data in the folder!" << endl;
        return;
    } else {
        cout << "[Main ][Info ] Read " << numImgs << " image files in the folder." << endl;
    }
}
