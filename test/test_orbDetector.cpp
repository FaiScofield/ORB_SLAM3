#include "ORBextractor.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#define ENABLE_SAVE_RESULT 1
// #define DEFAULT_DATASET_FOLDER "/home/vance/dataset/se2/DatasetRoom/image/"
// #define DEFAULT_OUTPUT_FOLDER "/home/vance/output/se2/"
#define DEFAULT_DATASET_FOLDER "/home/vance/dataset/fzu/201224_indoor/image/"
#define DEFAULT_OUTPUT_FOLDER "/home/vance/output/"
#define OUTPUT_FILE_PREFIX  "with_mask_"

using namespace ORB_SLAM3;
using namespace cv;
using namespace std;
namespace bf = boost::filesystem;

void readImagesRK(const string& strImagePath, vector<string>& vstrImages,
                  vector<double>& vTimeStamps);
void readImagesSE2(const string& strImagePath, vector<string>& vstrImages,
                   vector<double>& vTimeStamps);
void readImagesFZU(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps);


int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
                             "{inputFolder  i| |input data folder}"
                             "{outputFolder o| |output data folder}"
                             "{undistort    u|false|undistort image}"
                             "{equalize     e|false|equalize image histogram}"
                             "{removeOE     r|true |remove features on over exposure area}"
                             "{help         h|false|show help message}");

    if (/* argc < 2 ||  */ parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    String inputFolder = parser.get<String>("inputFolder");
    if (inputFolder.empty())
        inputFolder = string(DEFAULT_DATASET_FOLDER);
    cout << " - input data folder: " << inputFolder << endl;

    String outputFolder = parser.get<String>("outputFolder");
    if (outputFolder.empty())
        outputFolder = string(DEFAULT_OUTPUT_FOLDER);
    cout << " - output data folder: " << outputFolder << endl;

    bool bUndistortImg = parser.get<bool>("undistort");
    cout << " - bUndistortImg: " << bUndistortImg << endl;
    bool bEqualizeImg = parser.get<bool>("equalize");
    cout << " - bEqualizeImg: " << bEqualizeImg << endl;
    bool bRemoveOE = parser.get<bool>("removeOE");
    cout << " - bRemoveOE: " << bRemoveOE << endl;

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
    // readImagesSE2(inputFolder, vStrImages, vTimeStamps);
    readImagesFZU(inputFolder, vStrImages, vTimeStamps);

    ORBextractor detector(500, 2, 3, 17, 12);
    detector.SetFeatureType(ORBextractor::SURF);

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

    char waterlog[64], imgFile[64];
    Mat kernel1 = getStructuringElement(MORPH_RECT, Size(5,5));
    Mat kernel2 = getStructuringElement(MORPH_RECT, Size(15,15));
    Mat img, gray, outImg1, outImg2, grayEq;
    Mat descriptors, mask;
    vector<KeyPoint> vKPs1, vKPs2;
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
        claher->apply(gray, grayEq);

        if (bRemoveOE) {
            threshold(gray, mask, 200, 255, THRESH_BINARY_INV);
            dilate(mask, mask, kernel1);
            erode(mask, mask, kernel2);
        } else {
            mask = cv::noArray().getMat();
        }

        detector(gray, mask, vKPs1, descriptors, vLappingArea);
        drawKeypoints(gray, vKPs1, outImg1, Scalar(0, 255, 0));

        snprintf(waterlog, 64, "original image #%d, KPs:%ld", k, vKPs1.size());
        putText(outImg1, waterlog, Point(50,50), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,255));
        cout << "# " << k << " Number ORB features on original gray image: " << vKPs1.size() << endl;

        detector(grayEq, mask, vKPs2, descriptors, vLappingArea);
        drawKeypoints(grayEq, vKPs2, outImg2, Scalar(0, 255, 0));

        snprintf(waterlog, 64, "hist equalized #%d, KPs:%ld", k, vKPs2.size());
        putText(outImg2, waterlog, Point(50,50), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,255));
        cout << "# " << k << " Number ORB features on equalized gray image: " << vKPs2.size() << endl;

        Mat outImg;
        hconcat(outImg1, outImg2, outImg);
    #if ENABLE_SAVE_RESULT
        if (k % 10 == 0) {
            // snprintf(imgFile, 64, "%s/%sresult_no_eq_%04d.png", outputFolder.c_str(), OUTPUT_FILE_PREFIX, k);
            // imwrite(imgFile, outImg1);
            // snprintf(imgFile, 64, "%s/%sresult_with_eq_%04d.png", outputFolder.c_str(), OUTPUT_FILE_PREFIX, k);
            // imwrite(imgFile, outImg2);
            snprintf(imgFile, 64, "%s/%sresult_compare_%04d.png", outputFolder.c_str(), OUTPUT_FILE_PREFIX, k);
            imwrite(imgFile, outImg);
        }
    #endif
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
            // format: /frameRaw12987978101.png
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

// format: .../1596818919.30935264.png
void readImagesFZU(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps)
{
    bf::path path(strImagePath);
    if (!bf::exists(path)) {
        cerr << "[Main ][Error] Data folder doesn't exist!" << endl;
        return;
    }

    vector<pair<string, double>> vstrImgTime;
    vstrImgTime.reserve(3000);

    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;
        if (bf::is_regular_file(iter->status())) {
            // format: /frameRaw12987978101.png
            string s = iter->path().string();
            size_t i = s.find_last_of('/');
            size_t j = s.find_last_of('.');
            if (i == string::npos || j == string::npos)
                continue;
            auto t = atof(s.substr(i + 1, j - i - 1).c_str());
            vstrImgTime.emplace_back(s, t);
        }
    }
    int a = 1;
    a = a + 1;
    sort(vstrImgTime.begin(), vstrImgTime.end(),
         [](const pair<string, double>& lf, const pair<string, double>& rf) {
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
        vTimeStamps[k] = vstrImgTime[k].second;
    }
}