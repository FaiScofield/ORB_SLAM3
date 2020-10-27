#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "gms_matcher.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#define DEFAULT_DATASET_FOLDER "/home/vance/dataset/se2/DatasetRoom/image/"

#define IMAGE_WIDTH 752
#define IMAGE_HEIGHT 480
#define IMAGE_SUBREGION_NUM_COL 6
#define IMAGE_SUBREGION_NUM_ROW 4
#define IMAGE_BORDER 16
#define MAX_FEATURE_NUM 500
#define MAX_PYRAMID_LEVEL 3
#define PYRAMID_SCALE_FATOR 1.5

enum FeatureType {
    ORB = 0,
    FAST = 1,
    CV_FAST = 4,
    CV_ORB = 5,
    CV_SURF = 6,
    CV_SIFT = 7,
    CV_AKAZE = 8
};


using namespace std;
using namespace cv;
using namespace ORB_SLAM3;

void readImagesSE2(const string& strImagePath, vector<string>& vstrImages,
                   vector<double>& vTimeStamps);

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
                             "{type      t|ORB|value input type: ORB, CV_ORB, CV_SURF, CV_AKAZE}"
                             "{folder    f| |data folder}"
                             "{undistort u|false|undistort image}"
                             "{equalize  e|false|equalize image histogram}"
                             "{help      h|false|show help message}");

    if (argc < 2 || parser.get<bool>("help")) {
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

    /// read images
    vector<string> vImgFiles;
    vector<double> vTimeStamps;
    // cv::glob(str_folder, vImgFiles);
    readImagesSE2(strFolder, vImgFiles, vTimeStamps);
    const int nImgSize = vImgFiles.size();
    if (nImgSize == 0)
        return -1;
    // LOGI("Read " << nImgSize << " Images in the folder: " << str_folder);

    /// subregion params
    const int nImgWid = IMAGE_WIDTH;
    const int nImgHgt = IMAGE_HEIGHT;
    const int nImgBrd = IMAGE_BORDER;
    const int nSubregionNumCol = IMAGE_SUBREGION_NUM_COL;
    const int nSubregionNumRow = IMAGE_SUBREGION_NUM_ROW;
    const int nSubregionWid = (nImgWid - 2 * nImgBrd) / nSubregionNumCol;
    const int nSubregionHgt = (nImgHgt - 2 * nImgBrd) / nSubregionNumRow;
    vector<Rect> vSubregins(nSubregionNumCol * nSubregionNumRow);
    for (size_t r = 0; r < nSubregionNumRow; ++r) {
        const int y = nImgBrd + r * nSubregionHgt;
        int h = nSubregionHgt;
        if (r == nSubregionNumRow - 1) {
            h = max(h, nImgHgt - nImgBrd - y);
            // LOGT("Subregin height of row " << r << " is: " << h);
        }

        for (size_t c = 0; c < nSubregionNumCol; ++c) {
            const int idx = r * nSubregionNumCol + c;
            const int x = nImgBrd + c * nSubregionWid;
            int w = nSubregionWid;
            if (c == nSubregionNumCol - 1) {
                w = max(h, nImgWid - nImgBrd - x);
                // LOGT("Subregin width of col " << c << " is: " << w);
            }
            vSubregins[idx] = Rect(x, y, w, h);
        }
    }

    // auto pDetector = cv::ORB::create(MAX_FEATURE_NUM, PYRAMID_SCALE_FATOR, MAX_PYRAMID_LEVEL);
    Ptr<ORBextractor> pDetector = makePtr<ORBextractor>(ORBextractor(200, 1.5f, 3, 25, 15));
    Ptr<DescriptorMatcher> pMatcher = DescriptorMatcher::create("BruteForce-Hamming");
    Ptr<CLAHE> claher = createCLAHE(2.0, Size(6, 6));

    /// data
    size_t nKPs1, nKPs2;
    Mat image1, image2, imageOut;
    Mat descriptors1, descriptors2;
    vector<KeyPoint> vFeatures1, vFeatures2;
    vector<DMatch> vRoughMatches, vFineMatches;

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

    vector<int> overLapping{0, 0};
    for (int k = 0; k < nImgSize; ++k) {
        image2 = imread(vImgFiles[k], IMREAD_GRAYSCALE);
        if (image2.empty()) {
            // LOGW("Empty image #" << k << ": " << vImgFiles[k]);
            continue;
        }

        if (bUndistortImg) {
            Mat imgUn;
            cv::undistort(image2, imgUn, K, D);
            imgUn.copyTo(image2);
        }
        if (bEqualizeImg)
            claher->apply(image2, image2);

        (*pDetector)(image2, cv::noArray(), vFeatures2, descriptors2, overLapping);

        if (k > 0) {
            pMatcher->match(descriptors1, descriptors2, vRoughMatches);
            drawMatches(image1, vFeatures1, image2, vFeatures2, vRoughMatches, imageOut);
            imshow("Match BF", imageOut);
            waitKey(1);

            // gms
            std::vector<bool> vbInliers;
            GMS::gms_matcher gms(vFeatures1, image1.size(), vFeatures2, image2.size(), vRoughMatches);
            int num_inliers = gms.GetInlierMask(vbInliers, false, false);
            cout << "GMS Get total " << num_inliers << " matches." << endl;
            for (int i = 0; i < vbInliers.size(); ++i) {
                if (vbInliers[i])
                    vFineMatches.push_back(vRoughMatches[i]);
            }
            drawMatches(image1, vFeatures1, image2, vFeatures2, vFineMatches, imageOut);
            imshow("Match GMS", imageOut);
            waitKey(50);
        }

        // swap data
        image1 = image2.clone();
        descriptors1 = descriptors2.clone();
        vFeatures1.swap(vFeatures2);
    }

    return 0;
}

void readImagesSE2(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps)
{
    const size_t numImgs = 3108;
    vstrImages.resize(numImgs);
    vTimeStamps.resize(numImgs);
    double time = 6000000 * 1e-9;
    for (int i = 0; i < numImgs; ++i) {
        vstrImages[i] = strImagePath + "/" + to_string(i) + ".bmp";
        vTimeStamps[i] = time;
        time += 0.03 * 1e-9;
    }

    cout << "[Main ][Info ] Read " << numImgs << " image files in the folder." << endl;
}