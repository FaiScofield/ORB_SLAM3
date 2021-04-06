/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"

#include "Converter.h"
#include "G2oTypes.h"
#include "GeometricCamera.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "LineExtractor.h"
#include "LSDmatcher.h"
#include "LineIterator.h"

#include "CameraModels/KannalaBrandt8.h"
#include "CameraModels/Pinhole.h"
#include <thread>

#define ENABLE_DEBUG_FRAME 0

#if ENABLE_DEBUG_FRAME
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
#endif

namespace ORB_SLAM3
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

//For stereo fisheye matching
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false)
{
#if WITH_LINES
    mpLSDextractorLeft = NULL;
#endif
}


//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpcpi(frame.mpcpi),mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
     mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
     mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
     mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
     monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
     mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
     mTlr(frame.mTlr.clone()), mRlr(frame.mRlr.clone()), mtlr(frame.mtlr.clone()), mTrl(frame.mTrl.clone()), mTimeStereoMatch(frame.mTimeStereoMatch), mTimeORB_Ext(frame.mTimeORB_Ext)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++){
            mGrid[i][j]=frame.mGrid[i][j];
            if(frame.Nleft > 0){
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);

    if(!frame.mVw.empty())
        mVw = frame.mVw.clone();

    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;

    mImgLeft = frame.mImgLeft;

#if WITH_ODOMETRY
    mOdom = frame.mOdom;
    mOdomCalib = frame.mOdomCalib;
#endif

#if WITH_LINES
    mpLSDextractorLeft = frame.mpLSDextractorLeft;
    mnScaleLevelsLine = frame.mnScaleLevelsLine;
    mfScaleFactorLine = frame.mfScaleFactorLine;
    mfLogScaleFactorLine = frame.mfLogScaleFactorLine;
    mvScaleFactorsLine = frame.mvScaleFactorsLine;
    mvInvScaleFactorsLine = frame.mvInvScaleFactorsLine;
    mvLevelSigma2Line = frame.mvLevelSigma2Line;
    mvInvLevelSigma2Line = frame.mvInvLevelSigma2Line;

    NL = frame.NL;
    mLdesc = frame.mLdesc.clone();
    mvKeylinesUn = frame.mvKeylinesUn;
    mvKeyLineFunctions = frame.mvKeyLineFunctions;
    mvbLineOutlier = frame.mvbLineOutlier;
    mvpMapLines = frame.mvpMapLines;

    for (int i = 0; i < FRAME_GRID_COLS; i++)
    for (int j = 0; j < FRAME_GRID_ROWS; j++)
        mGridForLine[i][j] = frame.mGridForLine[i][j];
#endif
}

// Constructor for stereo cameras.
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false),
     mpCamera(pCamera) ,mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,0,0);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,0,0);
    threadLeft.join();
    threadRight.join();
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoMatches();
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif


    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);



        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF)
    {
        if(!pPrevF->mVw.empty())
            mVw = pPrevF->mVw.clone();
    }
    else
    {
        mVw = cv::Mat::zeros(3,1,CV_32F);
    }

    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mTlr = cv::Mat(3,4,CV_32F);
    mTrl = cv::Mat(3,4,CV_32F);
    mvStereo3Dpoints = vector<cv::Mat>(0);
    monoLeft = -1;
    monoRight = -1;
}

// Constructor for RGB-D cameras.
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,0);
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mTlr = cv::Mat(3,4,CV_32F);
    mTrl = cv::Mat(3,4,CV_32F);
    mvStereo3Dpoints = vector<cv::Mat>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}

// Constructor for Monocular cameras.
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif

#if WITH_LINES
    // undistort image
    cv::Mat mUndistX, mUndistY;
    initUndistortRectifyMap(mK, mDistCoef, cv::Mat_<double>::eye(3, 3), mK, cv::Size(imGray.cols, imGray.rows), CV_32F, mUndistX, mUndistY);
    cv::remap(imGray, mImgLeftUn, mUndistX, mUndistY, cv::INTER_LINEAR);

    // Scale Level Info for line
    assert(mpLSDextractorLeft);
    mnScaleLevelsLine = mpLSDextractorLeft->GetLevels();
    mfScaleFactorLine = mpLSDextractorLeft->GetScaleFactor();
    mfLogScaleFactorLine = log(mfScaleFactor);
    mvScaleFactorsLine = mpLSDextractorLeft->GetScaleFactors();
    mvInvScaleFactorsLine = mpLSDextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2Line = mpLSDextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2Line = mpLSDextractorLeft->GetInverseScaleSigmaSquares();

    thread threadPoint(&Frame::ExtractORB, this, 0, mImgLeftUn, 0, 1000);
    thread threadLine(&Frame::ExtractLSD, this, mImgLeftUn, cv::Mat());
    threadPoint.join();
    threadLine.join();
    
    N = mvKeys.size();
    NL = mvKeylinesUn.size();  //特征线的数量
    if (mvKeys.empty())
        return;
    mvKeysUn = mvKeys;

    mvpMapLines = vector<MapLine*>(NL, static_cast<MapLine*>(NULL));
    mvbLineOutlier = vector<bool>(NL, false);
#else
    ExtractORB(0, imGray, 0, 1000);
    N = mvKeys.size();
    if (mvKeys.empty())
        return;
    UndistortKeyPoints();
#endif

#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    mImgLeft = imGray.clone();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }


    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mTlr = cv::Mat(3,4,CV_32F);
    mTrl = cv::Mat(3,4,CV_32F);
    mvStereo3Dpoints = vector<cv::Mat>(0);
    monoLeft = -1;
    monoRight = -1;

#if WITH_LINES
    thread threadAssignPoint(&Frame::AssignFeaturesToGrid, this);
    thread threadAssignLine(&Frame::AssignFeaturesToGridForLine, this);
    threadAssignPoint.join();
    threadAssignLine.join();
#else
    AssignFeaturesToGrid();
#endif

    // mVw = cv::Mat::zeros(3,1,CV_32F);
    if(pPrevF)
    {
        if(!pPrevF->mVw.empty())
            mVw = pPrevF->mVw.clone();
    }
    else
    {
        mVw = cv::Mat::zeros(3,1,CV_32F);
    }

    mpMutexImu = new std::mutex();
}

#if 0
// Constructor for Monocular cameras with mask
Frame::Frame(const cv::Mat& imGray, const cv::Mat& mask, const double& timeStamp,
             ORBextractor* extractor, ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat& distCoef,
             const float& bf, const float& thDepth, Frame* pPrevF, const IMU::Calib& ImuCalib):
    mpcpi(NULL),
    mpORBvocabulary(voc), mpORBextractorLeft(extractor),
    mpORBextractorRight(static_cast<ORBextractor*>(NULL)), mTimeStamp(timeStamp),
    mK(static_cast<Pinhole*>(pCamera)->toK()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
    mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL),
    mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera),
    mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0)
{
    // Frame ID
    mnId = nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    mMaskLeft = mask;   //.clone();
    mMaskRight = mask;  //.clone();
    ExtractORB(0, imGray, 0, 1000);
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndExtORB - time_StartExtORB)
                       .count();
#endif


    N = mvKeys.size();
    if (mvKeys.empty())
        return;

    mImgLeft = imGray.clone();

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N, -1);
    mvDepth = vector<float>(N, -1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(N, static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();  // = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N, false);

    // This is done only for the first Frame (or after a change in the calibration)
    if (mbInitialComputations) {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0, 0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1, 1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0, 2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

    mb = mbf / fx;

    // Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mTlr = cv::Mat(3, 4, CV_32F);
    mTrl = cv::Mat(3, 4, CV_32F);
    mvStereo3Dpoints = vector<cv::Mat>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    // mVw = cv::Mat::zeros(3,1,CV_32F);
    if (pPrevF) {
        if (!pPrevF->mVw.empty())
            mVw = pPrevF->mVw.clone();
    } else {
        mVw = cv::Mat::zeros(3, 1, CV_32F);
    }

    mpMutexImu = new std::mutex();
}
#endif

#if WITH_ODOMETRY
// Constructor for Monocular-Odometry
Frame::Frame(const cv::Mat& imGray, const cv::Mat& mask, double timeStamp, ORBextractor* extractor, 
             ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat& distCoef, const float& bf, const float& thDepth, 
             Frame* pPrevF, const ODOM::Calib& odomCalib):
    mpcpi(NULL),
    mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
    mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mDistCoef(distCoef.clone()), mbf(bf),
    mThDepth(thDepth), mImuCalib(IMU::Calib()), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),
    mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false),
    mpCamera(pCamera), mpCamera2(nullptr), mTimeStereoMatch(0), mTimeORB_Ext(0),
    mOdomCalib(odomCalib)/* , mpOdomPreintegrated(NULL), mpOdomPreintegratedFrame(NULL) */
{
    // Frame ID
    mnId = nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    mMaskLeft = mask;   //.clone();
    mMaskRight = mask;  //.clone();

    // ORB extraction
#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif

#if WITH_LINES
    // undistort image
    cv::Mat mUndistX, mUndistY;
    initUndistortRectifyMap(mK, mDistCoef, cv::Mat_<double>::eye(3, 3), mK, cv::Size(imGray.cols, imGray.rows), CV_32F, mUndistX, mUndistY);
    cv::remap(imGray, mImgLeftUn, mUndistX, mUndistY, cv::INTER_LINEAR);

    // Scale Level Info for line
    assert(mpLSDextractorLeft);
    mnScaleLevelsLine = mpLSDextractorLeft->GetLevels();
    mfScaleFactorLine = mpLSDextractorLeft->GetScaleFactor();
    mfLogScaleFactorLine = log(mfScaleFactor);
    mvScaleFactorsLine = mpLSDextractorLeft->GetScaleFactors();
    mvInvScaleFactorsLine = mpLSDextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2Line = mpLSDextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2Line = mpLSDextractorLeft->GetInverseScaleSigmaSquares();

    thread threadPoint(&Frame::ExtractORB, this, 0, mImgLeftUn, 0, 1000);
    thread threadLine(&Frame::ExtractLSD, this, mImgLeftUn, cv::noArray());
    threadPoint.join();
    threadLine.join();
    
    N = mvKeys.size();
    NL = mvKeylinesUn.size();  //特征线的数量
    if (mvKeys.empty())
        return;
    mvKeysUn = mvKeys;

    mvpMapLines = vector<MapLine*>(NL, static_cast<MapLine*>(NULL));
    mvbLineOutlier = vector<bool>(NL, false);
#else
    ExtractORB(0, imGray, 0, 1000);
    N = mvKeys.size();
    if (mvKeys.empty())
        return;
    UndistortKeyPoints();
#endif

#ifdef SAVE_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_EndExtORB - time_StartExtORB)
                       .count();
#endif

    mImgLeft = imGray.clone();

    // Set no stereo information
    mvuRight = vector<float>(N, -1);
    mvDepth = vector<float>(N, -1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(N, static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();  // = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N, false);

    // This is done only for the first Frame (or after a change in the calibration)
    if (mbInitialComputations) {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0, 0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1, 1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0, 2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }


    mb = mbf / fx;

    // Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mTlr = cv::Mat(3, 4, CV_32F);
    mTrl = cv::Mat(3, 4, CV_32F);
    mvStereo3Dpoints = vector<cv::Mat>(0);
    monoLeft = -1;
    monoRight = -1;

#if WITH_LINES
    thread threadAssignPoint(&Frame::AssignFeaturesToGrid, this);
    thread threadAssignLine(&Frame::AssignFeaturesToGridForLine, this);
    threadAssignPoint.join();
    threadAssignLine.join();
#else
    AssignFeaturesToGrid();
#endif

    // mVw = cv::Mat::zeros(3,1,CV_32F);
    if (pPrevF) {
        if (!pPrevF->mVw.empty())
            mVw = pPrevF->mVw.clone();
    } else {
        mVw = cv::Mat::zeros(3, 1, CV_32F);
    }

    mpMutexOdom = new std::mutex();
}

// ODOM pose
cv::Mat Frame::GetOdomPosition()
{
    return mRwc * mOdomCalib.Tcb.rowRange(0, 3).col(3) + mOw;
}

cv::Mat Frame::GetOdomRotation()
{
    return mRwc * mOdomCalib.Tcb.rowRange(0, 3).colRange(0, 3);
}

cv::Mat Frame::GetOdomPose()
{
    cv::Mat Twb = cv::Mat::eye(4, 4, CV_32F);
    Twb.rowRange(0, 3).colRange(0, 3) = mRwc * mOdomCalib.Tcb.rowRange(0, 3).colRange(0, 3);
    Twb.rowRange(0, 3).col(3) = mRwc * mOdomCalib.Tcb.rowRange(0, 3).col(3) + mOw;
    return Twb.clone();
}
#endif  // IF WITH_ODOMETRY

void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

    int nReserve = 0.5f*N/(nCells);

    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].reserve(nReserve);
            if(Nleft != -1){
                mGridRight[i][j].reserve(nReserve);
            }
        }

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                                 : (i < Nleft) ? mvKeys[i]
                                                                 : mvKeysRight[i - Nleft];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            if(Nleft == -1 || i < Nleft)
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }

#if ENABLE_DEBUG_FRAME && 0
    // draw grid points
    Mat tmpShow = mImgLeft.clone();
    resize(tmpShow, tmpShow, Size(mImgLeft.cols*2, mImgLeft.rows*2));
    if (tmpShow.channels() == 1) {
        cvtColor(tmpShow, tmpShow, COLOR_GRAY2BGR);
    }
    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++) {
        const float cell_tl_x = i / mfGridElementWidthInv;
        const float cell_ct_x = (i + 0.5) / mfGridElementWidthInv;
        line(tmpShow, Point(cell_tl_x*2, mnMinY*2), Point(cell_tl_x*2, mnMaxY*2), Scalar(255, 0, 0));
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++) {
            const float cell_tl_y = j / mfGridElementHeightInv;
            const float cell_ct_y = (j + 0.5) / mfGridElementHeightInv;
            line(tmpShow, Point(mnMinX*2, cell_tl_y*2), Point(mnMaxX*2, cell_tl_y*2), Scalar(255, 0, 0));
            putText(tmpShow, to_string(mGrid[i][j].size()), Point(cell_ct_x*2, cell_ct_y*2),
                    FONT_HERSHEY_COMPLEX_SMALL, 0.5, Scalar(0, 0, 255), 1);
        }
    }
    imshow("AssignFeaturesToGrid", tmpShow);
    waitKey(1);
#endif
}

void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0,x1};
    if(flag==0)
        // monoLeft = (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors, vLapping);
        monoLeft = (*mpORBextractorLeft)(im, mMaskLeft, mvKeys, mDescriptors, vLapping);
    else
        // monoRight = (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight, vLapping);
        monoRight = (*mpORBextractorRight)(im, mMaskRight, mvKeysRight, mDescriptorsRight, vLapping);

#if ENABLE_DEBUG_FRAME && 0
    // draw all KPs
    cv::Mat tmpShow = im;
    // if (!mMaskLeft.empty()) {
    //     cv::bitwise_and(im, mMaskLeft, tmpShow);
    // }
    cv::drawKeypoints(tmpShow, mvKeys, tmpShow, cv::Scalar(0, 255, 0));
    cv::imshow("ExtractORB", tmpShow);
    cv::waitKey(1);
#endif
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::GetPose(cv::Mat &Tcw)
{
    Tcw = mTcw.clone();
}

void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

void Frame::SetVelocity(const cv::Mat &Vwb)
{
    mVw = Vwb.clone();
}

void Frame::SetImuPoseVelocity(const cv::Mat &Rwb, const cv::Mat &twb, const cv::Mat &Vwb)
{
    mVw = Vwb.clone();
    cv::Mat Rbw = Rwb.t();
    cv::Mat tbw = -Rbw*twb;
    cv::Mat Tbw = cv::Mat::eye(4,4,CV_32F);
    Rbw.copyTo(Tbw.rowRange(0,3).colRange(0,3));
    tbw.copyTo(Tbw.rowRange(0,3).col(3));
    mTcw = mImuCalib.Tcb*Tbw;
    UpdatePoseMatrices();
}



void Frame::UpdatePoseMatrices()
{
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

cv::Mat Frame::GetImuPosition()
{
    return mRwc*mImuCalib.Tcb.rowRange(0,3).col(3)+mOw;
}

cv::Mat Frame::GetImuRotation()
{
    return mRwc*mImuCalib.Tcb.rowRange(0,3).colRange(0,3);
}

cv::Mat Frame::GetImuPose()
{
    cv::Mat Twb = cv::Mat::eye(4,4,CV_32F);
    Twb.rowRange(0,3).colRange(0,3) = mRwc*mImuCalib.Tcb.rowRange(0,3).colRange(0,3);
    Twb.rowRange(0,3).col(3) = mRwc*mImuCalib.Tcb.rowRange(0,3).col(3)+mOw;
    return Twb.clone();
}


bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    if(Nleft == -1){
        // cout << "\na";
        pMP->mbTrackInView = false;
        pMP->mTrackProjX = -1;
        pMP->mTrackProjY = -1;

        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();

        // cout << "b";

        // 3D in camera coordinates
        const cv::Mat Pc = mRcw*P+mtcw;
        const float Pc_dist = cv::norm(Pc);

        // Check positive depth
        const float &PcZ = Pc.at<float>(2);
        const float invz = 1.0f/PcZ;
        if(PcZ<0.0f)
            return false;

        const cv::Point2f uv = mpCamera->project(Pc);

        // cout << "c";

        if(uv.x<mnMinX || uv.x>mnMaxX)
            return false;
        if(uv.y<mnMinY || uv.y>mnMaxY)
            return false;

        // cout << "d";
        pMP->mTrackProjX = uv.x;
        pMP->mTrackProjY = uv.y;

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const cv::Mat PO = P-mOw;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            return false;

        // cout << "e";

        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();

        // cout << "f";

        const float viewCos = PO.dot(Pn)/dist;

        if(viewCos<viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // cout << "g";

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = uv.x;
        pMP->mTrackProjXR = uv.x - mbf*invz;

        pMP->mTrackDepth = Pc_dist;
        // cout << "h";

        pMP->mTrackProjY = uv.y;
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        // cout << "i";

        return true;
    }
    else{
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP -> mnTrackScaleLevel = -1;
        pMP -> mnTrackScaleLevelR = -1;

        pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    // cout << "c";

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

cv::Mat Frame::inRefCoordinates(cv::Mat pCw)
{
    return mRcw*pCw+mtcw;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    /*cout << "fX " << factorX << endl;
    cout << "fY " << factorY << endl;*/

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

#if ENABLE_DEBUG_FRAME && 0
    // draw features in the area
    Mat tmpShow = mImgLeft.clone();
    if (tmpShow.channels() == 1) {
        cvtColor(tmpShow, tmpShow, COLOR_GRAY2BGR);
    }
    for (int x = nMinCellX; x <= nMaxCellX; ++x) {
        line(tmpShow, Point(x / mfGridElementWidthInv, nMinCellY / mfGridElementHeightInv),
             Point(x / mfGridElementWidthInv, nMaxCellY / mfGridElementHeightInv), Scalar(255, 0, 0));
    }
    for (int y = nMinCellY; y <= nMaxCellY; ++y) {
        line(tmpShow, Point(nMinCellX / mfGridElementWidthInv, y / mfGridElementHeightInv),
             Point(nMaxCellX / mfGridElementWidthInv, y / mfGridElementHeightInv), Scalar(255, 0, 0));
    }
    for (auto it = mvKeysUn.begin(); it != mvKeysUn.end(); ++it) {
        circle(tmpShow, (*it).pt, 1, Scalar(255,0,0), -1);
    }
    circle(tmpShow, Point(x, y), r, Scalar(0, 0, 255), 2);
    for (int i = 0; i < vIndices.size(); ++i) {
        circle(tmpShow, mvKeysUn[vIndices[i]].pt, 1, Scalar(0,255,0), -1);
    }
    putText(tmpShow, to_string(vIndices.size()), Point(x, y), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255), 1);
    imshow("GetFeaturesInArea", tmpShow);
    waitKey(200);
#endif

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);

    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);


    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }

#if ENABLE_DEBUG_FRAME
    // draw undistorted KPs
    cv::Mat imgUn, tmpShow1, tmpShow2, tmpShow3;
    cv::drawKeypoints(mImgLeft, mvKeys, tmpShow1, cv::Scalar(0, 255, 0));
    cv::undistort(mImgLeft, imgUn, static_cast<Pinhole*>(mpCamera)->toK(), mDistCoef);
    cv::drawKeypoints(imgUn, mvKeysUn, tmpShow2, cv::Scalar(0, 255, 0));
    cv::hconcat(tmpShow1, tmpShow2, tmpShow3);
    cv::imshow("Undistor Keypoints", tmpShow3);
    cv::waitKey(1);
#endif
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            //IL.convertTo(IL,CV_32F);
            //IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);
            IL.convertTo(IL,CV_16S);
            IL = IL - IL.at<short>(w,w);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                //IR.convertTo(IR,CV_32F);
                //IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);
                IR.convertTo(IR,CV_16S);
                IR = IR - IR.at<short>(w,w);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

bool Frame::imuIsPreintegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

// Constructor Stereo FishEye
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, cv::Mat& Tlr,Frame* pPrevF, const IMU::Calib &ImuCalib)
        :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2), mTlr(Tlr)
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    mImgLeft = imLeft.clone();
    mImgRight = imRight.clone();
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
    threadLeft.join();
    threadRight.join();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    Nleft = mvKeys.size();
    Nright = mvKeysRight.size();
    N = Nleft + Nright;

    if(N == 0)
        return;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf / fx;

    mRlr = mTlr.rowRange(0,3).colRange(0,3);
    mtlr = mTlr.col(3);

    cv::Mat Rrl = mTlr.rowRange(0,3).colRange(0,3).t();
    cv::Mat trl = Rrl * (-1 * mTlr.col(3));

    cv::hconcat(Rrl,trl,mTrl);

    ComputeStereoFishEyeMatches();
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    //Put all descriptors in the same matrix
    cv::vconcat(mDescriptors,mDescriptorsRight,mDescriptors);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();

    mpMutexImu = new std::mutex();

    UndistortKeyPoints();
    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    double t_read = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count();
    double t_orbextract = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
    double t_stereomatches = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t3 - t2).count();
    double t_assign = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t4 - t3).count();
    double t_undistort = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t5 - t4).count();

    /*cout << "Reading time: " << t_read << endl;
    cout << "Extraction time: " << t_orbextract << endl;
    cout << "Matching time: " << t_stereomatches << endl;
    cout << "Assignment time: " << t_assign << endl;
    cout << "Undistortion time: " << t_undistort << endl;*/

}

void Frame::ComputeStereoFishEyeMatches() {
    //Speed it up by matching keypoints in the lapping area
    vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    mvLeftToRightMatch = vector<int>(Nleft,-1);
    mvRightToLeftMatch = vector<int>(Nright,-1);
    mvDepth = vector<float>(Nleft,-1.0f);
    mvuRight = vector<float>(Nleft,-1);
    mvStereo3Dpoints = vector<cv::Mat>(Nleft);
    mnCloseMPs = 0;

    //Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

    int nMatches = 0;
    int descMatches = 0;

    //Check matches using Lowe's ratio
    for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it){
        if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7){
            //For every good match, check parallax and reprojection error to discard spurious matches
            cv::Mat p3D;
            descMatches++;
            float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
            float depth = static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2,mvKeys[(*it)[0].queryIdx + monoLeft],mvKeysRight[(*it)[0].trainIdx + monoRight],mRlr,mtlr,sigma1,sigma2,p3D);
            if(depth > 0.0001f){
                mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D.clone();
                mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                nMatches++;
            }
        }
    }
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos();

    cv::Mat mR, mt, twc;
    if(bRight){
        cv::Mat Rrl = mTrl.colRange(0,3).rowRange(0,3);
        cv::Mat trl = mTrl.col(3);
        mR = Rrl * mRcw;
        mt = Rrl * mtcw + trl;
        twc = mRwc * mTlr.rowRange(0,3).col(3) + mOw;
    }
    else{
        mR = mRcw;
        mt = mtcw;
        twc = mOw;
    }

    // 3D in camera coordinates
    cv::Mat Pc = mR*P+mt;
    const float Pc_dist = cv::norm(Pc);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    cv::Point2f uv;
    if(bRight) uv = mpCamera2->project(Pc);
    else uv = mpCamera->project(Pc);

    if(uv.x<mnMinX || uv.x>mnMaxX)
        return false;
    if(uv.y<mnMinY || uv.y>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-twc;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    if(bRight){
        pMP->mTrackProjXR = uv.x;
        pMP->mTrackProjYR = uv.y;
        pMP->mnTrackScaleLevelR= nPredictedLevel;
        pMP->mTrackViewCosR = viewCos;
        pMP->mTrackDepthR = Pc_dist;
    }
    else{
        pMP->mTrackProjX = uv.x;
        pMP->mTrackProjY = uv.y;
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackDepth = Pc_dist;
    }

    return true;
}

cv::Mat Frame::UnprojectStereoFishEye(const int &i){
    return mRwc*mvStereo3Dpoints[i]+mOw;
}

int Frame::CountObservations() const
{
    int ret = 0;
    for (MapPoint* pMP : mvpMapPoints) {
        if (pMP && !pMP->isBad())
            ret++;
    }
    return ret;
}

#if WITH_LINES
void Frame::AssignFeaturesToGridForLine()
{
    int nReserve = 0.5f*NL/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGridForLine[i][j].reserve(nReserve);

    // 在mGrid中记录了各特征点
    //#pragma omp parallel for
    for(int i=0;i<NL;i++)
    {
        const cv::KeyLine &kl = mvKeylinesUn[i];

        list<pair<int, int>> line_coords;

        LineIterator* it = new LineIterator(kl.startPointX * mfGridElementWidthInv, kl.startPointY * mfGridElementHeightInv, kl.endPointX * mfGridElementWidthInv, kl.endPointY * mfGridElementHeightInv);

        std::pair<int, int> p;
        while (it->getNext(p))
            if (p.first >= 0 && p.first < FRAME_GRID_COLS && p.second >= 0 && p.second < FRAME_GRID_ROWS)
                mGridForLine[p.first][p.second].push_back(i);

        delete [] it;
    }
}

void Frame::ExtractLSD(const cv::Mat &im, const cv::Mat &mask)
{
    (*mpLSDextractorLeft)(im, mask, mvKeylinesUn, mLdesc, mvKeyLineFunctions);
}


// 根据两个匹配的特征线计算特征线的3D坐标, frame1是当前帧，frame2是前一帧
void Frame::ComputeLine3D(Frame &frame1, Frame &frame2)
{
    //-------------------------计算两帧的匹配线段-----------------------------
    cv::BFMatcher* bfm = new cv::BFMatcher(cv::NORM_HAMMING, false);
    cv::Mat ldesc1, ldesc2;
    vector<vector<cv::DMatch>> lmatches;
    vector<cv::DMatch> good_matches;
    ldesc1 = frame1.mLdesc;
    ldesc2 = frame2.mLdesc;
    bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);
    // sort matches by the distance between the best and second best matches
    double nn_dist_th, nn12_dist_th;
    frame1.lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
//    nn12_dist_th = nn12_dist_th * 0.1;
        nn12_dist_th = nn12_dist_th * 0.5;
    sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
    vector<cv::KeyLine> keylines1 = frame1.mvKeylinesUn;     //暂存mvKeylinesUn的集合
    frame1.mvKeylinesUn.clear();    //清空当前帧的mvKeylinesUn
    vector<cv::KeyLine> keylines2;
    for(size_t i=0; i<lmatches.size(); i++)
    {
        double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
        if(dist_12 > nn12_dist_th)
        {
            //认为这个匹配比较好，应该更新该帧的的匹配线
            good_matches.push_back(lmatches[i][0]);
            frame1.mvKeylinesUn.push_back(keylines1[lmatches[i][0].queryIdx]);  //更新当前帧的匹配线
            keylines2.push_back(frame2.mvKeylinesUn[lmatches[i][0].trainIdx]);  //暂存前一帧的匹配线，用于计算3D端点
        }
    }
    //-------------------计算当前帧mvKeylinesUn对应的3D端点---------------------
    ///-step 1：frame1的R,t，世界坐标系，相机内参
    //-step 1.1:先获取frame1的R,t
    cv::Mat Rcw1 = frame1.mRcw;     //world to camera
    cv::Mat Rwc1 = frame1.mRwc;
    cv::Mat tcw1 = frame1.mtcw;
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    //-step 1.2：得到当前帧在世界坐标系中的坐标
    cv::Mat Ow1 = frame1.mOw;

    //-step 1.3:获取frame1的相机内参
    const float &fx1 = frame1.fx;
    const float &fy1 = frame1.fy;
    const float &cx1 = frame1.cx;
    const float &cy1 = frame1.cy;
    const float &invfx1 = frame1.invfx;
    const float &invfy1 = frame1.invfy;

    ///-step 2: frame2的R,t，世界坐标系，相机内参
    //-step 2.1：获取R，t
    cv::Mat Rcw2 = frame2.mRcw;
    cv::Mat Rwc2 = frame2.mRwc;
    cv::Mat tcw2 = frame2.mtcw;
    cv::Mat Tcw2(3, 4, CV_32F);
    Rcw2.copyTo(Tcw2.colRange(0,3));
    tcw2.copyTo(Tcw2.col(3));

    //-step 2.2：获取frame2在世界坐标系中的坐标
    cv::Mat Ow2 = frame2.mOw;

    //-step 2.3：获取frame2的相机内参
    const float &fx2 = frame2.fx;
    const float &fy2 = frame2.fy;
    const float &cx2 = frame2.cx;
    const float &cy2 = frame2.cy;
    const float &invfx2 = frame2.invfx;
    const float &invfy2 = frame2.invfy;

    ///-step 3:对每对匹配通过三角化生成3D端点
    cv::Mat vBaseline = Ow2 - Ow1;  //frame1和frame2的位移
    const float baseline = norm(vBaseline);

    //-step 3.1:根据两帧的姿态计算两帧之间的基本矩阵, Essential Matrix: t12叉乘R2
    const Mat &K1 = frame1.mK;
    const Mat &K2 = frame2.mK;
    cv::Mat R12 = Rcw1*Rwc2;
    cv::Mat t12 = -Rcw1*Rwc2*tcw2 + tcw1;
    cv::Mat t12x = SkewSymmetricMatrix(t12);
    cv::Mat essential_matrix = K1.t().inv()*t12x*R12*K2.inv();

    //-step3.2：三角化
       const int nlmatches = good_matches.size();
        for (int i = 0; i < nlmatches; ++i)
        {
            cv::KeyLine &kl1 = frame1.mvKeylinesUn[i];
            cv::KeyLine &kl2 = keylines2[i];

            //------起始点,start points-----
            // 得到特征线段的起始点在归一化平面上的坐标
            cv::Mat sn1 = (Mat_<float>(3,1) << (kl1.startPointX-cx1)*invfx1, (kl1.startPointY-cy1)*invfy1, 1.0);
            cv::Mat sn2 = (Mat_<float>(3,1) << (kl2.startPointX-cx2)*invfx2, (kl2.startPointY-cy2)*invfy2, 1.0);

            // 把对应的起始点坐标转换到世界坐标系下
            cv::Mat sray1 = Rwc1 * sn1;
            cv::Mat sray2 = Rwc2 * sn2;
            // 计算在世界坐标系下，两个坐标向量间的余弦值
            const float cosParallax_sn = sray1.dot(sray2)/(norm(sray1) * norm(sray2));
            cv::Mat s3D;
            if(cosParallax_sn > 0 && cosParallax_sn < 0.998)
            {
                // linear triangulation method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = sn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = sn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = sn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = sn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                Mat w1, u1, vt1;
                SVD::compute(A, w1, u1, vt1, SVD::MODIFY_A|SVD::FULL_UV);

                s3D = vt1.row(3).t();

                if(s3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                s3D = s3D.rowRange(0,3)/s3D.at<float>(3);
            }
            cv::Mat s3Dt = s3D.t();

            //-------结束点,end points-------
            // 得到特征线段的起始点在归一化平面上的坐标
            cv::Mat en1 = (Mat_<float>(3,1) << (kl1.endPointX-cx1)*invfx1, (kl1.endPointY-cy1)*invfy1, 1.0);
            cv::Mat en2 = (Mat_<float>(3,1) << (kl2.endPointX-cx2)*invfx2, (kl2.endPointY-cy2)*invfy2, 1.0);

            // 把对应的起始点坐标转换到世界坐标系下
            cv::Mat eray1 = Rwc1 * en1;
            cv::Mat eray2 = Rwc2 * en2;
            // 计算在世界坐标系下，两个坐标向量间的余弦值
            const float cosParallax_en = eray1.dot(eray2)/(norm(eray1) * norm(eray2));
            cv::Mat e3D;
            if(cosParallax_en > 0 && cosParallax_en < 0.998)
            {
                // linear triangulation method
                cv::Mat B(4,4,CV_32F);
                B.row(0) = en1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                B.row(1) = en1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                B.row(2) = en2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                B.row(3) = en2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w2, u2, vt2;
                cv::SVD::compute(B, w2, u2, vt2, cv::SVD::MODIFY_A|cv::SVD::FULL_UV);

                e3D = vt2.row(3).t();

                if(e3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                e3D = e3D.rowRange(0,3)/e3D.at<float>(3);
            }
            cv::Mat e3Dt = e3D.t();

            //-step 3.3：检测生成的3D点是否在相机前方，两个帧都需要检测
            float sz1 = Rcw1.row(2).dot(s3Dt) + tcw1.at<float>(2);
            if(sz1<=0)
                continue;

            float sz2 = Rcw2.row(2).dot(s3Dt) + tcw2.at<float>(2);
            if(sz2<=0)
                continue;

            float ez1 = Rcw1.row(2).dot(e3Dt) + tcw1.at<float>(2);
            if(ez1<=0)
                continue;

            float ez2 = Rcw2.row(2).dot(e3Dt) + tcw2.at<float>(2);
            if(ez2<=0)
                continue;

            //生成特征点时还有检测重投影误差和检测尺度连续性两个步骤，但是考虑到线特征的特殊性，先不写这两步
            //MapLine(int idx_, Vector6d line3D_, Mat desc_, int kf_obs_, Vector3d obs_, Vector3d dir_, Vector4d pts_, double sigma2_=1.f);
//            MapLine* pML = new MapLine();

        }

}

// line descriptor MAD, 自己添加的
void Frame::lineDescriptorMAD( vector<vector<cv::DMatch>> line_matches, double &nn_mad, double &nn12_mad) const
{
    vector<vector<cv::DMatch>> matches_nn, matches_12;
    matches_nn = line_matches;
    matches_12 = line_matches;
//    cout << "Frame::lineDescriptorMAD——matches_nn = "<<matches_nn.size() << endl;

    // estimate the NN's distance standard deviation
    double nn_dist_median;
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    nn_dist_median = matches_nn[int(matches_nn.size()/2)][0].distance;

    for(unsigned int i=0; i<matches_nn.size(); i++)
        matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);
    sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    nn_mad = 1.4826 * matches_nn[int(matches_nn.size()/2)][0].distance;

    // estimate the NN's 12 distance standard deviation
    double nn12_dist_median;
    sort( matches_12.begin(), matches_12.end(), conpare_descriptor_by_NN12_dist());
    nn12_dist_median = matches_12[int(matches_12.size()/2)][1].distance - matches_12[int(matches_12.size()/2)][0].distance;
    for (unsigned int j=0; j<matches_12.size(); j++)
        matches_12[j][0].distance = fabsf( matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);
    sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
    nn12_mad = 1.4826 * matches_12[int(matches_12.size()/2)][0].distance;
}


/**
 * @brief 判断MapLine的两个端点是否在视野内
 *
 * @param pML               MapLine
 * @param viewingCosLimit   视角和平均视角的方向阈值
 * @return                  true if the MapLine is in view
 */
bool Frame::isInFrustum(MapLine *pML, float viewingCosLimit)
{
    pML->mbTrackInView = false;

    Vector6d P = pML->GetWorldPos();

    cv::Mat SP = (Mat_<float>(3,1) << P(0), P(1), P(2));
    cv::Mat EP = (Mat_<float>(3,1) << P(3), P(4), P(5));

    // 两个端点在相机坐标系下的坐标
    const cv::Mat SPc = mRcw*SP + mtcw;
    const float &SPcX = SPc.at<float>(0);
    const float &SPcY = SPc.at<float>(1);
    const float &SPcZ = SPc.at<float>(2);

    const cv::Mat EPc = mRcw*EP + mtcw;
    const float &EPcX = EPc.at<float>(0);
    const float &EPcY = EPc.at<float>(1);
    const float &EPcZ = EPc.at<float>(2);

    // 检测两个端点的Z值是否为正
    if(SPcZ<0.0f || EPcZ<0.0f)
        return false;

    // V-D 1) 将端点投影到当前帧上，并判断是否在图像内
    const float invz1 = 1.0f/SPcZ;
    const float u1 = fx * SPcX * invz1 + cx;
    const float v1 = fy * SPcY * invz1 + cy;

    if(u1<mnMinX || u1>mnMaxX)
        return false;
    if(v1<mnMinY || v1>mnMaxY)
        return false;

    const float invz2 = 1.0f/EPcZ;
    const float u2 = fx*EPcX*invz2 + cx;
    const float v2 = fy*EPcY*invz2 + cy;

    if(u2<mnMinX || u2>mnMaxX)
        return false;
    if(v2<mnMinY || v2>mnMaxY)
        return false;

    // V-D 3)计算MapLine到相机中心的距离，并判断是否在尺度变化的距离内
    const float maxDistance = pML->GetMaxDistanceInvariance();
    const float minDistance = pML->GetMinDistanceInvariance();
    // 世界坐标系下，相机到线段中点的向量，向量方向由相机指向中点
    const cv::Mat OM = 0.5*(SP+EP) - mOw;
    const float dist = cv::norm(OM);

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    // V-D 2)计算当前视角和平均视角夹角的余弦值，若小于cos(60°),即夹角大于60°则返回
    Eigen::Vector3d Pn = pML->GetNormal();
    cv::Mat pn = (Mat_<float>(3,1) << Pn(0), Pn(1), Pn(2));
    const float viewCos = OM.dot(pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    // V-D 4) 根据深度预测尺度（对应特征在一层）
    const int nPredictedLevel = pML->PredictScale(dist, mfLogScaleFactor);

    // Data used by the tracking
    // 标记该特征将来要被投影
    pML->mbTrackInView = true;
    pML->mTrackProjX1 = u1;
    pML->mTrackProjY1 = v1;
    pML->mTrackProjX2 = u2;
    pML->mTrackProjY2 = v2;
    pML->mnTrackScaleLevel = nPredictedLevel;
    pML->mTrackViewCos = viewCos;

    return true;
}


vector<size_t> Frame::GetFeaturesInAreaForLine(const float &x1, const float &y1, const float &x2, const float &y2, const float  &r, const int minLevel, const int maxLevel,const float TH) const
{
    vector<size_t> vIndices;
    vIndices.reserve(NL);
    unordered_set<size_t> vIndices_set;

    float x[3] = {x1, (x1+x2)/2.0f, x2};
    float y[3] = {y1, (y1+y2)/2.0f, y2}; 

    float delta1x = x1-x2;
    float delta1y = y1-y2;
    float norm_delta1 = sqrt(delta1x*delta1x + delta1y*delta1y);
    delta1x /= norm_delta1;
    delta1y /= norm_delta1;

    for(int i = 0; i<3;i++){
        const int nMinCellX = max(0,(int)floor((x[i]-mnMinX-r)*mfGridElementWidthInv));
        if(nMinCellX>=FRAME_GRID_COLS)
            continue;

        const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x[i]-mnMinX+r)*mfGridElementWidthInv));
        if(nMaxCellX<0)
            continue;

        const int nMinCellY = max(0,(int)floor((y[i]-mnMinY-r)*mfGridElementHeightInv));
        if(nMinCellY>=FRAME_GRID_ROWS)
            continue;

        const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y[i]-mnMinY+r)*mfGridElementHeightInv));
        if(nMaxCellY<0)
            continue;

        for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
        {
            for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
            {
                const vector<size_t> vCell = mGridForLine[ix][iy];
                if(vCell.empty())
                    continue;

                for(size_t j=0, jend=vCell.size(); j<jend; j++)
                {
                    if(vIndices_set.find(vCell[j]) != vIndices_set.end())
                        continue;

                    const KeyLine &klUn = mvKeylinesUn[vCell[j]];

                    float delta2x = klUn.startPointX - klUn.endPointX;
                    float delta2y = klUn.startPointY - klUn.endPointY;
                    float norm_delta2 = sqrt(delta2x*delta2x + delta2y*delta2y);
                    delta2x /= norm_delta2;
                    delta2y /= norm_delta2;
                    float CosSita = abs(delta1x * delta2x + delta1y * delta2y);

                    if(CosSita < TH)
                        continue;

                    Eigen::Vector3d Lfunc = mvKeyLineFunctions[vCell[j]]; 
                    const float dist = Lfunc(0)*x[i] + Lfunc(1)*y[i] + Lfunc(2);

                    if(fabs(dist)<r)
                    {
                        if(vIndices_set.find(vCell[j]) == vIndices_set.end())
                        {
                            vIndices.push_back(vCell[j]);
                            vIndices_set.insert(vCell[j]);
                        }
                    }
                }
            }
        }
    }
    
    return vIndices;
}

vector<size_t> Frame::GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2, const float &r,
                                     const int minLevel, const int maxLevel, const float TH) const
{
    vector<size_t> vIndices;

    vector<cv::KeyLine> vkl = this->mvKeylinesUn;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>0);

    float delta1x = x1-x2;
    float delta1y = y1-y2;
    float norm_delta1 = sqrt(delta1x*delta1x + delta1y*delta1y);
    delta1x /= norm_delta1;
    delta1y /= norm_delta1;

    for(size_t i=0; i<vkl.size(); i++)
    {
        cv::KeyLine keyline = vkl[i];

        // 1.对比中点距离
        float distance = (0.5*(x1+x2)-keyline.pt.x)*(0.5*(x1+x2)-keyline.pt.x)+(0.5*(y1+y2)-keyline.pt.y)*(0.5*(y1+y2)-keyline.pt.y);
        if(distance > r*r)
            continue;

        float delta2x = vkl[i].startPointX - vkl[i].endPointX;
        float delta2y = vkl[i].startPointY - vkl[i].endPointY;
        float norm_delta2 = sqrt(delta2x*delta2x + delta2y*delta2y);
        delta2x /= norm_delta2;
        delta2y /= norm_delta2;
        float CosSita = abs(delta1x * delta2x + delta1y * delta2y);

        if(CosSita < TH)
            continue;

        // 3.比较金字塔层数
        if(bCheckLevels)
        {
            if(keyline.octave<minLevel)
                continue;
            if(maxLevel>=0 && keyline.octave>maxLevel)
                continue;
        }

        vIndices.push_back(i);
    }

    return vIndices;
}


#endif

}  // namespace ORB_SLAM3
