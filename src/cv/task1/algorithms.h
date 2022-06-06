#ifndef CGCV_ALGORITHMS_H
#define CGCV_ALGORITHMS_H

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class algorithms {
public:


  static void manual_filter_2d(const Mat &input, Mat &output, const Mat &kernel);

  static void
  initializer_kernel_matrix(int radius, double sigma, Mat &k_L, Mat &k_R, Mat &k, Mat &k_gauss_2d, Mat &k_gauss_1d);

  static Mat apply_separable_conv2d(const Mat &input, const Mat &y_kernel, const Mat &x_kernel);

  static void sw_filter(Mat input, const Mat &kernel, const Mat &kernel_left, const Mat &kernel_right, Mat &result,
                        vector<vector<Mat>> &abs_differences_channel);

  static void bgr_to_yuv(Mat bgr_image, Mat &yuv_image);

  static void yuv_to_bgr(Mat yuv_image, Mat &bgr_image);

  static void visualize_yuv_channels(const Mat& yuv_image, const std::vector<std::vector<int>> u_color_map,
      const std::vector<std::vector<int>> v_color_map, std::vector<Mat>& yuv_visualized_images);

  static void bonus(Mat &input, const Mat &kernel, const Mat &kernel_left, const Mat& kernel_right, Mat &output);

  static void create_bilateral_masks(const Mat &kernel, const Mat &kernel_left, const Mat &kernel_right, vector<Mat> &kernels);

  static void bilateral_filter(const Mat &input, const Mat &kernel, Mat &output);

  static float compute_bilateral_weight(const Mat &input, const int i, const int j, const int k, const int l, const float sigma_d);

  static void find_best_fitting_side_window(vector<Mat> &d_abs, Mat &channel, vector<Mat> &filtered_imgs);

  static void calc_abs_difference(const vector<Mat>& d, vector<Mat>& d_abs, const Mat& input);
};


#endif //CGCV_ALGORITHMS_H
