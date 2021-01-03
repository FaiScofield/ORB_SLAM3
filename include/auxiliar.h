
#ifndef AUXILIAR_H
#define AUXILIAR_H

#include <math.h>

#if 0

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <vector>

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

// 比较线特征距离的两种方式，自己添加的
struct compare_descriptor_by_NN_dist {
    inline bool operator()(const std::vector<cv::DMatch>& a, const std::vector<cv::DMatch>& b)
    {
        return (a[0].distance < b[0].distance);
    }
};

struct conpare_descriptor_by_NN12_dist {
    inline bool operator()(const std::vector<cv::DMatch>& a, const std::vector<cv::DMatch>& b)
    {
        return ((a[1].distance - a[0].distance) > (b[1].distance - b[0].distance));
    }
};

// 按描述子之间距离的从小到大方式排序
struct sort_descriptor_by_queryIdx {
    inline bool operator()(const std::vector<cv::DMatch>& a, const std::vector<cv::DMatch>& b)
    {
        return (a[0].queryIdx < b[0].queryIdx);
    }
};

struct sort_lines_by_response {
    inline bool operator()(const cv::line_descriptor::KeyLine& a, const cv::line_descriptor::KeyLine& b)
    {
        return (a.response > b.response);
    }
};

inline cv::Mat SkewSymmetricMatrix(const cv::Mat& v)
{
    return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
        v.at<float>(2), 0, -v.at<float>(0),
        -v.at<float>(1), v.at<float>(0), 0);
}

/**
 * @brief 求一个vector数组的中位数绝对偏差MAD
 * 中位数绝对偏差MAD——median absolute deviation, 是单变量数据集中样本差异性的稳健度量。
 * MAD是一个健壮的统计量，对于数据集中异常值的处理比标准差更具有弹性，可以大大减少异常值对于数据集的影响
 * 对于单变量数据集 X={X1,X2,X3,...,Xn}, MAD的计算公式为：MAD(X)=median(|Xi-median(X)|)
 * @param residues
 * @return
 */
inline double vector_mad(std::vector<double> residues)
{
    if (residues.size() != 0) {
        // Return the standard deviation of std::vector with MAD estimation
        int n_samples = residues.size();
        sort(residues.begin(), residues.end());
        double median = residues[n_samples / 2];
        for (int i = 0; i < n_samples; i++)
            residues[i] = fabs(residues[i] - median);
        sort(residues.begin(), residues.end());
        double MAD = residues[n_samples / 2];
        return 1.4826 * MAD;
    } else
        return 0.0;
}
#endif

// LOG define
#include <iostream>

#define LOGF(msg) (std::cerr << "\033[41m-- |F| " << msg << "\033[0m" << std::endl)
#define LOGE(msg) (std::cerr << "\033[31m-- |E| " << msg << "\033[0m" << std::endl)
#define LOGW(msg) (std::cerr << "\033[33m-- |W| " << msg << "\033[0m" << std::endl)
#define LOGI(msg) (std::cout << "\033[00m-- |I| " << msg << "\033[0m" << std::endl)
#define LOGD(msg) (std::cout << "\033[34m-- |D| " << msg << "\033[0m" << std::endl)
#define LOGT(msg) (std::cout << "\033[32m-- |T| " << msg << "\033[0m" << std::endl)

#define K_LOGE(id, msg) (std::cerr << "\033[31m-- |E| #" << id << " " << msg << "\033[0m" << std::endl)
#define K_LOGW(id, msg) (std::cerr << "\033[33m-- |W| #" << id << " " << msg << "\033[0m" << std::endl)
#define K_LOGI(id, msg) (std::cout << "\033[00m-- |I| #" << id << " " << msg << "\033[0m" << std::endl)
#define K_LOGD(id, msg) (std::cout << "\033[34m-- |D| #" << id << " " << msg << "\033[0m" << std::endl)
#define K_LOGT(id, msg) (std::cout << "\033[32m-- |T| #" << id << " " << msg << "\033[0m" << std::endl)

// -pi ~ +pi
inline double NormalizeAngle(double theta)
{
#if 0
    return theta + 2*M_PI*floor((M_PI - theta)/(2*M_PI));
#else
    if (theta >= -M_PI && theta < M_PI)
        return theta;

    double multiplier = floor(theta / (2 * M_PI));
    theta = theta - multiplier * 2 * M_PI;
    if (theta >= M_PI)
        theta -= 2 * M_PI;
    if (theta < -M_PI)
        theta += 2 * M_PI;

    return theta;
#endif
}
#endif  // AUXILIAR_H