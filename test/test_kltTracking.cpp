#include "ORBextractor.h"
#include "gms_matcher.h"

#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <vector>

#define ENABLE_SAVE_RESULT 1
#define DEFAULT_DATASET_FOLDER "/home/vance/dataset/se2/DatasetRoom/image/"
#define DEFAULT_OUTPUT_FOLDER "/home/vance/output/se2/"

using namespace ORB_SLAM3;
using namespace cv;
using namespace std;
namespace bf = boost::filesystem;

string g_vocabularyFile = "/home/vance/slam_ws/ORB_SLAM3/Vocabulary/ORBvoc.bin";

void readImagesRK(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps);
void readImagesSE2(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps);

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
                             "{inputFolder  i| |input data folder}"
                             "{outputFolder o| |output data folder}"
                             "{undistort    u|false|undistort image}"
                             "{equalize     e|true |equalize image histogram}"
                             "{removeOE     r|true |remove features on over exposure area}"
                             "{gms          g|false|use GMS to filter matches instead of check orientation}"
                             "{help         h|false|show help message}");

    if (/* argc < 2 || */ parser.get<bool>("help")) {
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
    bool bUseGMS = parser.get<bool>("gms");
    cout << " - bUseGMS: " << bUseGMS << endl;


    /// read data
    vector<string> vStrImages;
    vector<double> vTimeStamps;
#if 1
    readImagesSE2(inputFolder, vStrImages, vTimeStamps);
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
#endif

    /// classes
    Ptr<CLAHE> pClaher = createCLAHE(2.0, Size(6, 6));
    Ptr<ORBextractor> pDetector = makePtr<ORBextractor>(ORBextractor(500, 2, 3, 25, 15));


    Mat imgRef, imgCur, grayRef, grayCur, grayEq, mask, outImg;
    vector<Point2f> vKPsRef, vKPsCur, vPrevMatched;
    vector<int> vMatches12;
    vector<DMatch> vDMatches, vDMatchesRefine;
    const Mat kernel1 = getStructuringElement(MORPH_RECT, Size(5, 5));
    const Mat kernel2 = getStructuringElement(MORPH_RECT, Size(15, 15));

    int nBaseIdx = 0;
    long long aveMatches[10] = {0};
    for (int k = 0, kend = vStrImages.size(); k < kend; ++k) {
        cout << "#" << k << " Dealing with img " << vStrImages[k] << endl;

        imgCur = imread(vStrImages[k], IMREAD_COLOR);
        if (imgCur.empty()) {
            cerr << "Open image error: " << vStrImages[k] << endl;
            continue;
        }

        cvtColor(imgCur, grayCur, COLOR_BGR2GRAY);
        if (bUndistortImg) {
            Mat imgUn;
            cv::undistort(grayCur, imgUn, K, D);
            imgUn.copyTo(grayCur);
            cv::undistort(imgCur, imgUn, K, D);
            imgUn.copyTo(imgCur);
        }

        if (bRemoveOE) {
            threshold(grayCur, mask, 200, 255, THRESH_BINARY_INV);
            dilate(mask, mask, kernel1);
            erode(mask, mask, kernel2);
            mask.rowRange(0, 20).setTo(0);
            mask.rowRange(mask.rows - 20, mask.rows).setTo(0);
            mask.colRange(0, 20).setTo(0);
            mask.colRange(mask.cols - 20, mask.cols).setTo(0);
        } else {
            mask = cv::noArray().getMat();
        }

        if (bEqualizeImg) {
            pClaher->apply(grayCur, grayEq);
            grayEq.copyTo(grayCur);
        }

        // base frame
        if (k % 8 == 0) {
            nBaseIdx = k;
            goodFeaturesToTrack(grayCur, vKPsRef, 300, 0.01, 15, mask);
            imgRef = imgCur.clone();
            grayRef = grayCur.clone();
            vPrevMatched = vKPsRef;
            continue;
        }

        if (k > 0) {
            Mat tmpShow = imgCur.clone();
            vector<KeyPoint> tmpKP;
            KeyPoint::convert(vPrevMatched, tmpKP);
            drawKeypoints(imgCur, tmpKP, tmpShow, Scalar(255,0,0));


            vector<uchar> vTrackFlags;
            vector<float> vTrackErrors;
            calcOpticalFlowPyrLK(grayRef, grayCur, vPrevMatched, vKPsCur, vTrackFlags, vTrackErrors, Size(32,32));

            KeyPoint::convert(vKPsCur, tmpKP);
            drawKeypoints(tmpShow, tmpKP, tmpShow, Scalar(0,255,0));
            imshow("KPs prev(blue) & curr(gren)", tmpShow);
            waitKey(10);

            vDMatches.clear();
            vDMatches.reserve(vTrackFlags.size());
            int nMatches = 0;
            for (size_t i = 0; i < vTrackFlags.size(); ++i) {
                if (vTrackFlags[i]) {
                    vDMatches.emplace_back(i, i, 1);
                    nMatches++;
                }
            }
            if (nMatches < 5) {
                cerr << "Too less matches! " << nMatches << endl;
                continue;
            }

            vector<KeyPoint> vKPs1, vKPs2;
            KeyPoint::convert(vKPsRef, vKPs1);
            KeyPoint::convert(vKPsCur, vKPs2);
            drawMatches(imgRef, vKPs1, imgCur, vKPs2, vDMatches, outImg, Scalar(255, 0, 0));

            // gms
            if (bUseGMS) {
                std::vector<bool> vbInliers;
                GMS::gms_matcher gms(vKPs1, grayRef.size(), vKPs2, grayCur.size(), vDMatches);
                nMatches = gms.GetInlierMask(vbInliers, false, true);
                cout << "GMS Get total " << nMatches << " matches." << endl;

                vDMatchesRefine.clear();
                vDMatchesRefine.reserve(vDMatches.size());
                for (int i = 0; i < vbInliers.size(); ++i) {
                    if (vbInliers[i]) {
                        vDMatchesRefine.push_back(vDMatches[i]);
                    }
                }
                drawMatches(imgRef, vKPs1, imgCur, vKPs2, vDMatchesRefine, outImg, Scalar(0, 255, 0));
                vDMatches.swap(vDMatchesRefine);
            }

            aveMatches[k - nBaseIdx - 1] += nMatches;
            const int nAveMatches = aveMatches[k - nBaseIdx - 1] / (nBaseIdx / 10 + 1);
            cout << "#" << k << " to #" << nBaseIdx << ", matches: " << nMatches
                 << ", ave matches: " << nAveMatches << endl;

            char waterMark[64], imgFile[128];
            snprintf(waterMark, 64, "Idx: %d - %d, matches: %d, ave: %d", nBaseIdx, k, nMatches, nAveMatches);
            putText(outImg, waterMark, Point(50, 50), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
            imshow("KLT features matches", outImg);
            waitKey(1);
            snprintf(imgFile, 128, "%s/KLT_matches_%03d_to_%03d.jpg", outputFolder.c_str(), nBaseIdx, k);
            imwrite(imgFile, outImg);

            // Warp
            if (vDMatches.size() > 4) {
                vector<Point2f> vFeatures1, vFeatures2;
                vFeatures1.reserve(vDMatches.size());
                vFeatures2.reserve(vDMatches.size());
                for (DMatch& m : vDMatches) {
                    vFeatures1.push_back(vKPsRef[m.queryIdx]);
                    vFeatures2.push_back(vKPsCur[m.trainIdx]);
                }
                Mat H, A;
                vector<uchar> inliers;
                H = findHomography(vFeatures2, vFeatures1, RANSAC, 3, inliers);
                A = estimateAffinePartial2D(vFeatures2, vFeatures1, inliers, RANSAC);
//                H = findHomography(vFeatures2, vFeatures1, LMEDS, 3, inliers);
//                A = estimateAffinePartial2D(vFeatures2, vFeatures1, inliers, LMEDS);

                Mat blendOut, blendH, blendA;
                Mat warpH, warpA;
                if (!H.empty())
                    warpPerspective(imgCur, warpH, H, imgCur.size());
                else
                    warpH = Mat::zeros(imgCur.size(), imgCur.depth());
                if (!A.empty())
                    warpAffine(imgCur, warpA, A, imgCur.size());
                else
                    warpA = Mat::zeros(imgCur.size(), imgCur.depth());
                addWeighted(imgRef, 0.5, warpH, 0.5, 0, blendH);
                addWeighted(imgRef, 0.5, warpA, 0.5, 0, blendA);
                hconcat(blendH, blendA, blendOut);
                imshow("Warpe Idmages H/A", blendOut);
                waitKey(100);
            }
        }   // if k

        // swap keyframe
        vPrevMatched = vKPsCur;
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
    const size_t numImgs = 3108;
    vstrImages.reserve(numImgs);
    vTimeStamps.reserve(numImgs);
    double time = 6000000 * 1e-9;
    for (int i = 220; i < numImgs; ++i) {
        vstrImages.push_back(strImagePath + "/" + to_string(i) + ".bmp");
        vTimeStamps.push_back(time);
        time += 0.03 * 1e-9;
    }

    cout << "[Main ][Info ] Read " << vstrImages.size() << " image files in the folder." << endl;
}
