#include "CameraModels/Pinhole.h"
#include "Frame.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "gms_matcher.h"

#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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
    ORBextractor detector(300, 2, 3, 11, 7);
    Ptr<CLAHE> claher = createCLAHE(2.0, Size(6, 6));

    vector<float> vCamCalib{207.9359613169054, 207.4159055585876, 160.5827136112504, 117.7328673795551};
    Ptr<GeometricCamera> pCamera = Ptr<GeometricCamera>(dynamic_cast<GeometricCamera*>(new Pinhole(vCamCalib)));
    Ptr<ORBextractor> pDetector = makePtr<ORBextractor>(ORBextractor(300, 2, 3, 11, 7));
    Ptr<ORBmatcher> pMatcher = makePtr<ORBmatcher>(ORBmatcher(0.6, true));
    if (bUseGMS) {
        pMatcher->setCheckOrientation(false);
    }
    Ptr<ORBVocabulary> pVocabulary = makePtr<ORBVocabulary>(ORBVocabulary());
    if (!pVocabulary->loadFromBinaryFile(g_vocabularyFile)) {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << g_vocabularyFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    Frame frameCur, frameRef;
    Mat img, gray, grayEq, mask, outImg;
    vector<Point2f> vPrevMatched;
    vector<int> vMatches12;
    vector<DMatch> vDMatches, vDMatchesRefine;
    vector<int> vLappingArea = {0, 1000};
    const Mat kernel1 = getStructuringElement(MORPH_RECT, Size(5, 5));
    const Mat kernel2 = getStructuringElement(MORPH_RECT, Size(15, 15));

    int nBaseIdx = 0;
    long long aveMatches[10] = {0};
    for (int k = 0, kend = vStrImages.size(); k < kend; ++k) {
        cout << "#" << k << " Dealing with img " << vStrImages[k] << endl;

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

        if (bRemoveOE) {
            threshold(gray, mask, 200, 255, THRESH_BINARY_INV);
            dilate(mask, mask, kernel1);
            erode(mask, mask, kernel2);
        } else {
            mask = cv::noArray().getMat();
        }

        if (bEqualizeImg) {
            claher->apply(gray, grayEq);
            grayEq.copyTo(gray);
        }

        // frame construction
        Mat noDistort = cv::Mat::zeros(4, 1, CV_32FC1);
        frameCur = Frame(gray, mask, vTimeStamps[k], pDetector.get(), pVocabulary.get(),
                            pCamera.get(), noDistort, 0.f, 0.f);
        frameCur.imgLeft = gray.clone();

        int nMatches;
        if (k > 0) {
            nMatches = pMatcher->SearchForInitialization(frameRef, frameCur, vPrevMatched, vMatches12, 50);

            vDMatches.clear();
            vDMatches.reserve(vMatches12.size());
            for (size_t i = 0; i < vMatches12.size(); ++i) {
                if (vMatches12[i] >= 0 && vMatches12[i] < (int)frameCur.mvKeys.size()) {
                    vDMatches.emplace_back(i, vMatches12[i], 1);
                }
            }

            hconcat(frameRef.imgLeft, frameCur.imgLeft, outImg);
            drawMatches(frameRef.imgLeft, frameRef.mvKeys, frameCur.imgLeft, frameCur.mvKeys,
                        vDMatches, outImg, Scalar(255, 0, 0));

            // gms
            if (bUseGMS) {
                std::vector<bool> vbInliers;
                GMS::gms_matcher gms(frameRef.mvKeysUn, gray.size(), frameCur.mvKeysUn, gray.size(), vDMatches);
                nMatches = gms.GetInlierMask(vbInliers, false, true);
                cout << "GMS Get total " << nMatches << " matches." << endl;

                vDMatchesRefine.clear();
                vDMatchesRefine.reserve(vDMatches.size());
                for (int i = 0; i < vbInliers.size(); ++i) {
                    if (vbInliers[i]) {
                        vDMatchesRefine.push_back(vDMatches[i]);
                    }
                }
                drawMatches(frameRef.imgLeft, frameRef.mvKeys, frameCur.imgLeft, frameCur.mvKeys,
                            vDMatchesRefine, outImg, Scalar(0, 255, 0));
                vDMatches.swap(vDMatchesRefine);
            }

            aveMatches[k - nBaseIdx - 1] += nMatches;
            const int nAveMatches = aveMatches[k - nBaseIdx - 1] / (nBaseIdx / 10 + 1);
            cout << "#" << nBaseIdx << " to #" << k << ", matches: " << nMatches
                 << ", ave matches: " << nAveMatches << endl;

            char waterMark[64];
            snprintf(waterMark, 64, "Idx: %d - %d, matches: %d, ave: %d", nBaseIdx, k, nMatches, nAveMatches);
            putText(outImg, waterMark, Point(50, 50), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
            imshow("ORB features matches", outImg);
            waitKey(1);

            // Warp
            if (vDMatches.size() > 4) {

                vector<Point2f> vFeatures1, vFeatures2;
                vFeatures1.reserve(vDMatches.size());
                vFeatures2.reserve(vDMatches.size());
                for (DMatch& m : vDMatches) {
                    vFeatures1.push_back(frameRef.mvKeysUn[m.queryIdx].pt);
                    vFeatures2.push_back(frameCur.mvKeysUn[m.trainIdx].pt);
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
                    warpPerspective(gray, warpH, H, gray.size());
                else
                    warpH = Mat::zeros(gray.size(), gray.depth());
                if (!A.empty())
                    warpAffine(gray, warpA, A, gray.size());
                else
                    warpA = Mat::zeros(gray.size(), gray.depth());
                addWeighted(frameRef.imgLeft, 0.5, warpH, 0.5, 0, blendH);
                addWeighted(frameRef.imgLeft, 0.5, warpA, 0.5, 0, blendA);
                hconcat(blendH, blendA, blendOut);
                imshow("Warpe Idmages H/A", blendOut);
                waitKey(100);
            }
        }

        if (k % 8 == 0 || nMatches < 10) {
            nBaseIdx = k;
            frameRef = frameCur;
            KeyPoint::convert(frameCur.mvKeys, vPrevMatched);
        }

        // swap keyframe
        KeyPoint::convert(frameCur.mvKeys, vPrevMatched);
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
