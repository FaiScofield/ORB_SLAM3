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

#define TEST_DATA_TYPE          2 // 0-2: RK, SE2, FZU
#define FEATURE_TYPE            2 // 0-2: ORB, GFTT, SURF
#define ENABLE_USE_ODOMETRY     1
#define ENABLE_SAVE_RESULT      1
#define DEFAULT_DATASET_FOLDER "/home/vance/dataset/fzu/201224_hall_1/image/"
#define DEFAULT_OUTPUT_FOLDER  "/home/vance/output/fzu/"
// #define DEFAULT_DATASET_FOLDER "/home/vance/dataset/se2/DatasetRoom/image/"
// #define DEFAULT_OUTPUT_FOLDER   "/home/vance/output/se2/"
#define LOOP_FRAMES             20

using namespace ORB_SLAM3;
using namespace cv;
using namespace std;
namespace bf = boost::filesystem;

string g_vocabularyFile = "/home/vance/slam_ws/ORB_SLAM3/Vocabulary/ORBvoc.bin";

void readImagesRK(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps);
void readImagesSE2(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps);
void readImagesFZU(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps);

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
                             "{inputFolder  i| |input data folder}"
                             "{outputFolder o| |output data folder}"
                             "{undistort    u|false|undistort image}"
                             "{equalize     e|false|equalize image histogram}"
                             "{removeOE     r|false|remove features on over exposure area}"
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
#if TEST_DATA_TYPE == 0
    readImagesRK(inputFolder, vStrImages, vTimeStamps);
#elif TEST_DATA_TYPE == 1
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

    vector<float> vCamCalib{231.976033627644090, 232.157224036901510, 326.923920970539310, 227.838488395348380};
    Ptr<GeometricCamera> pCamera = Ptr<GeometricCamera>(dynamic_cast<GeometricCamera*>(new Pinhole(vCamCalib)));
#else
    readImagesFZU(inputFolder, vStrImages, vTimeStamps);
    cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);
    cv::Mat D = cv::Mat::zeros(4, 1, CV_32FC1);
    K.at<float>(0, 0) = 573.3130;
    K.at<float>(1, 1) = 573.3899;
    K.at<float>(0, 2) = 321.0333;
    K.at<float>(1, 2) = 243.5320;
    D.at<float>(0, 0) = 0.1248;
    D.at<float>(1, 0) = -0.2051;
    D.at<float>(2, 0) = 0.0;
    D.at<float>(3, 0) = 0.0;
    cout << endl << "K = " << K << endl;
    cout << "D = " << D.t() << endl;

    vector<float> vCamCalib{573.3130, 573.3899, 321.0333, 243.5320};
    Ptr<GeometricCamera> pCamera = Ptr<GeometricCamera>(dynamic_cast<GeometricCamera*>(new Pinhole(vCamCalib)));
#endif

#if ENABLE_USE_ODOMETRY
    vector<ORB_SLAM3::ODOM::Point> vOdometries;
    vOdometries.reserve(vStrImages.size());
    float x, y, theta;
    double timestamp;
    string line;

    ifstream rec(inputFolder + "/../odom_sync.txt");
    if (!rec.is_open()) {
        cerr << "[Main ][Error] Please check if the file exists!" << inputFolder + "/../odom_sync.txt" << endl;
    } else {
        while (std::getline(rec, line), !line.empty()) {
            istringstream iss(line);
            iss >> timestamp >> x >> y >> theta; // [m],[rad]
            vOdometries.emplace_back(x, y, theta, timestamp);
            line.clear();
        }
    }
    int nOdoms = static_cast<int>(vOdometries.size());
    if (nOdoms < 1 || nOdoms < vStrImages.size()) {
        cerr << "ERROR: Failed to load odometries! nOdoms = " << nOdoms << endl;
        nOdoms = 0;
    }
#endif

    /// classes
    Ptr<CLAHE> claher = createCLAHE(2.0, Size(6, 6));
    Ptr<ORBextractor> pDetector = makePtr<ORBextractor>(ORBextractor(300, 2, 3, 21, 11));
    Ptr<ORBmatcher> pMatcher = makePtr<ORBmatcher>(ORBmatcher(0.8, true));
    pDetector->SetFeatureType(FEATURE_TYPE);
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

    Mat H, A;
    Frame frameCur, frameRef;
    Mat img, gray, grayEq, mask, outImg;
    vector<Point2f> vPrevMatched;
    vector<int> vMatches12;
    vector<DMatch> vDMatches, vDMatchesRefine;
    vector<int> vLappingArea = {0, 1000};
    const Mat kernel1 = getStructuringElement(MORPH_RECT, Size(5, 5));
    const Mat kernel2 = getStructuringElement(MORPH_RECT, Size(15, 15));

    int nBaseIdx = 0;
    long long aveMatches[LOOP_FRAMES] = {0};
    long long aveInliers[LOOP_FRAMES] = {0};
    long long aveCntTime[LOOP_FRAMES] = {0};
    for (int k = 0, kend = vStrImages.size(); k < kend; ++k) {
        // cout << "#" << k << " Dealing with img " << vStrImages[k] << endl;

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
        frameCur.mImgLeft = gray.clone();

        int nMatches, nInliers;
        if (k > 0) {
            nMatches = pMatcher->SearchForInitialization(frameRef, frameCur, vPrevMatched, vMatches12, 100);

            vDMatches.clear();
            vDMatches.reserve(vMatches12.size());
            for (size_t i = 0; i < vMatches12.size(); ++i) {
                if (vMatches12[i] >= 0 && vMatches12[i] < (int)frameCur.mvKeysUn.size()) {
                    vDMatches.emplace_back(i, vMatches12[i], 1);
                }
            }

            hconcat(frameRef.mImgLeft, frameCur.mImgLeft, outImg);
            cvtColor(outImg, outImg, COLOR_GRAY2BGR);
            // drawMatches(frameRef.mImgLeft, frameRef.mvKeysUn, frameCur.mImgLeft, frameCur.mvKeysUn,
            //             vDMatches, outImg, Scalar(255, 0, 0));
            const cv::Point2f offset(frameRef.mImgLeft.cols, 0);
            int nMatchCnt = 0;
            for (int i = 0; i < frameRef.N; ++i) {
                cv::Point2f pt1 = frameRef.mvKeysUn[i].pt;
                cv::Point2f pt2 = frameCur.mvKeysUn[i].pt + offset;
                cv::circle(outImg, pt1, 1, Scalar(0,0,255), 1, LINE_AA);
                cv::circle(outImg, pt2, 1, Scalar(0,0,255), 1, LINE_AA);

                if (vMatches12[i] < 0)
                    continue;

                pt1 = frameRef.mvKeysUn[i].pt;
                pt2 = frameCur.mvKeysUn[vMatches12[i]].pt + offset;
                cv::circle(outImg, pt1, 2, Scalar(0,0,255), 2, LINE_AA);
                cv::circle(outImg, pt2, 2, Scalar(0,0,255), 2, LINE_AA);
                cv::line(outImg, pt1, pt2, Scalar(255,0,0), 1, LINE_AA);
                nMatchCnt++;
            }
            assert(nMatchCnt == nMatchCnt);
            Mat outImg2 = outImg.clone();

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
                drawMatches(frameRef.mImgLeft, frameRef.mvKeysUn, frameCur.mImgLeft, frameCur.mvKeysUn,
                            vDMatchesRefine, outImg, Scalar(0, 255, 0));
                vDMatches.swap(vDMatchesRefine);
            }

            aveCntTime[k - nBaseIdx - 1]++;
            aveMatches[k - nBaseIdx - 1] += nMatches;
            const float nAveMatches = aveMatches[k - nBaseIdx - 1] * 1.f / aveCntTime[k - nBaseIdx - 1];
            cout << "#" << nBaseIdx << " to #" << k << ", matches: " << nMatches
                 << ", ave matches: " << nAveMatches << endl;

            char waterMark[256];
            // snprintf(waterMark, 256, "Idx: %d-%d, matches: %d, ave: %.2f", nBaseIdx, k, nMatches, nAveMatches);
            // putText(outImg, waterMark, Point(50, 50), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 0));
            // imshow("ORB features matches", outImg);
            // waitKey(50);

            // Warp & remove outliers
            if (vDMatches.size() > 4) {
                vector<pair<int, int>> vIndexes;
                vIndexes.reserve(vDMatches.size());

                vector<Point2f> vFeatures1, vFeatures2;
                vFeatures1.reserve(vDMatches.size());
                vFeatures2.reserve(vDMatches.size());
                for (DMatch& m : vDMatches) {
                    vFeatures1.push_back(frameRef.mvKeysUn[m.queryIdx].pt);
                    vFeatures2.push_back(frameCur.mvKeysUn[m.trainIdx].pt);
                    vIndexes.emplace_back(m.queryIdx, m.trainIdx);
                }
                
                vector<uchar> inliers;
                H = findHomography(vFeatures2, vFeatures1, RANSAC, 3, inliers);
                inliers.clear();
                A = estimateAffinePartial2D(vFeatures2, vFeatures1, inliers, RANSAC);
                // H = findHomography(vFeatures2, vFeatures1, LMEDS, 3, inliers);
                // A = estimateAffinePartial2D(vFeatures2, vFeatures1, inliers, LMEDS);
                // cout << "A = \n" << A << endl;

                nInliers = 0;
                for (int i = 0; i < inliers.size(); ++i) {
                    if (!inliers[i])
                        continue;
                    cv::Point2f pt1 = frameRef.mvKeysUn[vIndexes[i].first].pt;
                    cv::Point2f pt2 = frameCur.mvKeysUn[vIndexes[i].second].pt + offset;
                    cv::line(outImg2, pt1, pt2, Scalar(0,255,0), 1, LINE_AA);
                    nInliers++;
                }

                aveInliers[k - nBaseIdx - 1] += nInliers;
                const float nAveInliers = aveInliers[k - nBaseIdx - 1] * 1.f / aveCntTime[k - nBaseIdx - 1];
                printf("#%d to #%d, inliers / matches: %d / %d, ratio: %.2f, ave: %.2f / %.2f\n", 
                    nBaseIdx, k, nInliers, nMatches, nInliers*1.f/nMatches, nAveInliers, nAveMatches);
                // cout << "#" << nBaseIdx << " to #" << k << ", inliers: " << nInliers << ", (" << nInliers*1.f/nMatches 
                //      << "), ave inliers: " << nAveInliers << endl;

                snprintf(waterMark, 256, "Idx: %d-%d, inliers/matches: %d/%d, ratio: %.2f", nBaseIdx, k, nInliers, nMatches, nInliers*1.f/nMatches);
                putText(outImg2, waterMark, Point(50, 50), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 0));
                imshow("ORB features inliers", outImg2);
                snprintf(waterMark, 256, "%s/inlier_match_%d_to_%d.png", outputFolder.c_str(), nBaseIdx, k);
                // if (nBaseIdx >= 80)
                //     imwrite(waterMark, outImg2);
                waitKey(50);

                // Mat blendOut, blendH, blendA;
                // Mat warpH, warpA;
                // if (!H.empty())
                //     warpPerspective(gray, warpH, H, gray.size());
                // else
                //     warpH = Mat::zeros(gray.size(), gray.depth());
                // if (!A.empty())
                //     warpAffine(gray, warpA, A, gray.size());
                // else
                //     warpA = Mat::zeros(gray.size(), gray.depth());
                // addWeighted(frameRef.mImgLeft, 0.5, warpH, 0.5, 0, blendH);
                // addWeighted(frameRef.mImgLeft, 0.5, warpA, 0.5, 0, blendA);
                // hconcat(blendH, blendA, blendOut);
                // imshow("Warpe Images H/A", blendOut);
                // waitKey(50);
            }
        }

        // swap keyframe
        if (k % LOOP_FRAMES == 0 || nMatches < 10) {
            nBaseIdx = k;
            frameRef = frameCur;
            KeyPoint::convert(frameCur.mvKeysUn, vPrevMatched);
        }

        // swap vPrevMatched
    #if 0//ENABLE_USE_ODOMETRY
        if (nOdoms > 0) {
            assert(vOdometries.size() == vStrImages.size());

            const ORB_SLAM3::ODOM::Point& odomPoint1 = vOdometries[nBaseIdx];
            const ORB_SLAM3::ODOM::Point& odomPoint2 = vOdometries[k];

            // predictPointsAndImage
            const double angle = odomPoint2.data.z - odomPoint1.data.z;
            Point2f center;
            center.x = vCamCalib[2] - 0;  // cx - Tbc.tx
            center.y = vCamCalib[3] - 0;   // cy - Tbc.ty

            // double row = static_cast<double>(240);
            double row = img.rows;
            for (int i = 0, iend = frameCur.N; i < iend; i++) {
                double x1 = frameCur.mvKeysUn[i].pt.x;
                double y1 = row - frameCur.mvKeysUn[i].pt.y;
                double x2 = center.x;
                double y2 = row - center.y;
                double x = cvRound((x1 - x2) * cos(angle) - (y1 - y2) * sin(angle) + x2);
                double y = cvRound((x1 - x2) * sin(angle) + (y1 - y2) * cos(angle) + y2);
                y = row - y;

                vPrevMatched[i] = Point2f(x, y);
            }
        }
    #elif 1
        KeyPoint::convert(frameRef.mvKeysUn, vPrevMatched);
        if (!A.empty() && nBaseIdx != k) {
            cv::Mat A12;
            cv::invertAffineTransform(A, A12);
            cv::transform(vPrevMatched, vPrevMatched, A12);
        }
    #else
        // KeyPoint::convert(frameCur.mvKeysUn, vPrevMatched);    // 效果差,特征点起始位置改变
        KeyPoint::convert(frameRef.mvKeysUn, vPrevMatched);    // 效果相对更好
    #endif
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

// format: .../1596818919.30935264.jpg
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
            // format: /frameRaw12987978101.jpg
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
