#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "gms_matcher.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

#define ENABLE_SAVE_RESULT 1
#define DEFAULT_DATASET_FOLDER "/home/vance/dataset/se2/DatasetRoom/image/"
#define DEFAULT_OUTPUT_FOLDER "/home/vance/output/gms/"
#define OUTPUT_FILE_PREFIX  "no_mask_with_eq_"


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
                             "{inputFolder  i| |input data folder}"
                             "{outputFolder o| |output data folder}"
                             "{undistort u|false|undistort image}"
                             "{equalize  e|false|equalize image histogram}"
                             "{removeOE     r|true |remove features on over exposure area}"
                             "{help      h|false|show help message}");

    if (argc < 2 || parser.get<bool>("help")) {
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
    bool bSharpen = false;
    cout << " - bSharpen: " << bSharpen << endl;

    /// read images
    vector<string> vImgFiles;
    vector<double> vTimeStamps;
    // cv::glob(str_folder, vImgFiles);
    readImagesSE2(inputFolder, vImgFiles, vTimeStamps);
    const int nImgSize = vImgFiles.size();
    if (nImgSize == 0)
        return -1;

#if 0
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
#endif

    Ptr<ORBextractor> pDetector = makePtr<ORBextractor>(ORBextractor(500, 2.0f, 3, 25, 15));
    Ptr<cv::ORB> pCvORB = cv::ORB::create(500, 2.0f, 3, 25, 15);
    Ptr<DescriptorMatcher> pMatcher = DescriptorMatcher::create("BruteForce-Hamming");
    Ptr<CLAHE> claher = createCLAHE(2.0, Size(8, 8));

    /// data
    size_t nKPs1, nKPs2;
    Mat image1, image2, imageOut, mask;
    Mat descriptors1, descriptors2;
    vector<Point2f> vPoints2;
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

    int refIdx = 0;
    char waterlog[64], imgFile[64];
    Mat kernel1 = getStructuringElement(MORPH_RECT, Size(5,5));
    Mat kernel2 = getStructuringElement(MORPH_RECT, Size(15,15));
    vector<int> overLapping{0, 0};
    for (int k = 0; k < nImgSize; ++k) {
        image2 = imread(vImgFiles[k], IMREAD_GRAYSCALE);
        if (image2.empty()) {
            cerr << "Open image error: " << vImgFiles[k] << endl;
            continue;
        }

        float corner_th = 0.005;
        if (bEqualizeImg) {
            claher->apply(image2, image2);
            corner_th = 0.01;
        }

        if (bRemoveOE) {
            threshold(image2, mask, 200, 255, THRESH_BINARY_INV);
            dilate(mask, mask, kernel1);
            erode(mask, mask, kernel2);
            mask.rowRange(0, IMAGE_BORDER).setTo(0);
            mask.rowRange(mask.rows - IMAGE_BORDER, mask.rows).setTo(0);
            mask.colRange(0, IMAGE_BORDER).setTo(0);
            mask.colRange(mask.cols - IMAGE_BORDER, mask.cols).setTo(0);
        } else {
            mask = cv::noArray().getMat();
        }

        Mat tmpImg;
        if (bUndistortImg) {
            cv::undistort(image2, tmpImg, K, D);
            image2 = tmpImg;
        }
        if (bSharpen){
            imshow("sharp before", image2);
            waitKey(10);
            Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
            filter2D(image2, tmpImg, CV_32F, kernel);
            convertScaleAbs(tmpImg, image2);
//            addWeighted(image2, 2, tmpImg, -1, 0, image2);
            imshow("sharp after", image2);
            waitKey(10);
        }

        // (*pDetector)(image2, mask, vFeatures2, descriptors2, overLapping);
        goodFeaturesToTrack(image2, vPoints2, 300, corner_th, 15, mask, 3, false);
//        cornerSubPix(image2, vPoints2, Size(5,5), Size(-1,-1), TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 20, 1e-5));
        KeyPoint::convert(vPoints2, vFeatures2);
        pDetector->compute(image2, vFeatures2, descriptors2);

        if (k > 0) {
            pMatcher->match(descriptors1, descriptors2, vRoughMatches);
            drawMatches(image1, vFeatures1, image2, vFeatures2, vRoughMatches, imageOut);
            snprintf(waterlog, 64, "BF matches for #%d & #%d, %ld", refIdx, k, vRoughMatches.size());
            putText(imageOut, waterlog, Point(50,50), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,255));
            snprintf(imgFile, 64, "%s/%sBF_match_%04d.jpg", outputFolder.c_str(), OUTPUT_FILE_PREFIX, k);
            imwrite(imgFile, imageOut);
            imshow("Match BF", imageOut);
            waitKey(1);

            // gms
            std::vector<bool> vbInliers;
            GMS::gms_matcher gms(vFeatures1, image1.size(), vFeatures2, image2.size(), vRoughMatches);
            int num_inliers = gms.GetInlierMask(vbInliers, false, true);
            cout << "GMS Get total " << num_inliers << " matches." << endl;

            vFineMatches.clear();
            vFineMatches.reserve(vRoughMatches.size());
            for (int i = 0; i < vbInliers.size(); ++i) {
                if (vbInliers[i])
                    vFineMatches.push_back(vRoughMatches[i]);
            }
            drawMatches(image1, vFeatures1, image2, vFeatures2, vFineMatches, imageOut);
            snprintf(waterlog, 64, "GMS matches for #%d & #%d, %ld", refIdx, k, vFineMatches.size());
            putText(imageOut, waterlog, Point(50,50), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,255));
            snprintf(imgFile, 64, "%s/%sGMS_match_%04d.jpg", outputFolder.c_str(), OUTPUT_FILE_PREFIX, k);
            imwrite(imgFile, imageOut);
            imshow("Match GMS", imageOut);
            waitKey(50);
        }

        // swap data
        if (k % 10 == 0) {
            refIdx = k;
            image1 = image2.clone();
            descriptors1 = descriptors2.clone();
            vFeatures1.swap(vFeatures2);
        }
    }

    return 0;
}

void readImagesSE2(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps)
{
    const size_t numImgs = 3108;
    vstrImages.reserve(numImgs);
    vTimeStamps.reserve(numImgs);
    double time = 6000000 * 1e-9;
    for (int i = 200; i < numImgs; ++i) {
        vstrImages.push_back(strImagePath + "/" + to_string(i) + ".bmp");
        vTimeStamps.push_back(time);
        time += 0.03 * 1e-9;
    }

    cout << "[Main ][Info ] Read " << numImgs << " image files in the folder." << endl;
}
