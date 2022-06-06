#include "algorithms.h"
#include <math.h>


//================================================================================
// initializer_kernel_matrix()
//--------------------------------------------------------------------------------
// TODO:
//  - create the SWF kernel matrices based upon the given radius
//  - create the gaussian 1D- and 2D-kernels based on radius and sigma
//
// parameters:
//  - radius: radius of the kernels
//  - sigma: standard deviation of gauss distribution
//  - k_L: the left part of the seperated kernel
//  - k_R: the right part of the seperated kernel
//  - k: the median kernel
//  - k_gauss_2d: the 2D gaussian kernel
//  - k_gauss_1d: the 1D gaussion kernel
// return: void
//================================================================================

void
algorithms::initializer_kernel_matrix(const int radius, const double sigma, Mat &k_L, Mat &k_R, Mat &k, Mat &k_gauss_2d,
                                      Mat &k_gauss_1d) {
  // TODO put your code here
  const Mat zero = Mat::zeros(1,2*radius+1,CV_32FC1);
  for (int i = 0; i<radius+1; i++){
    k_L = zero;
    k_L.at<float>(i) = 1.0/(radius+1);
  }

  const Mat zero1 = Mat::zeros(1,2*radius+1,CV_32FC1);
  for (int n = 2*radius+1; n>radius-1; n--){
    k_R = zero1;
    k_R.at<float>(n) = 1.0/(radius+1);
  }

  const Mat zero2 = Mat::zeros(1,2*radius+1,CV_32FC1);
  for (int i = 0; i<2*radius+1; i++){
    k = zero2;
    k.at<float>(i) = 1.0/(2*radius+1);
  }
  const Mat zero3 = Mat::zeros(1,2*radius+1,CV_32FC1);
  for (int i = 0; i<2*radius+1; i++){
    k_gauss_1d = zero3;
    k_gauss_1d.at<float>(i) = (1.0/(cv::sqrt(2.0*M_PI)*sigma))*cv::exp((cv::pow((i-radius),2)*(-1))/(2.0*cv::pow(sigma,2)));
  }
  const Mat zero4 = Mat::zeros(2*radius+1,2*radius+1,CV_32FC1);
  k_gauss_2d = zero4;
  for (int y = 0; y < 2*radius+1; y++){
    for (int x = 0; x < 2*radius+1; x++){
      k_gauss_2d.at<float>(y,x) = (1.0/(2.0*M_PI*cv::pow(sigma,2))*cv::exp(((cv::pow((y-radius),2)
        + cv::pow((x-radius),2))*(-1))/(2.0*cv::pow(sigma,2))));
    }
  }
  //imshow("output",k_gauss_2d);
 // cv::waitKey(0);
  //std::cout << k_L << std::endl;
  //std::cout << k_R << std::endl;
  //std::cout << k << std::endl;
  //std::cout << k_gauss_1d << std::endl;
  //std::cout << k_gauss_2d << std::endl;

}


//================================================================================
// manual_filter_2d()
//--------------------------------------------------------------------------------
// TODO:
//  - manually implement the 2d convolution. DO NOT use the cv function filter2D!
//
// parameters:
//  - input: the input Image
//  - output: the resulting Image
//  - kernel: the kernel used for the convolution
// return: void
//================================================================================
float sumOfSum(const Mat &input1, const Mat &kernel, int y, int x, int rows, int cols, int pad){
  float x_sum_e = 0;
  float x_sum = 0;
  for (int ky = 0; ky < kernel.rows; ky++)
  {
    for (int kx = 0; kx < kernel.cols; kx++)
    {

      int ky_buffer = y + ky - ((kernel.rows - 1) / 2)+pad;
      int kx_buffer = x + kx - ((kernel.cols - 1) / 2)+pad;
      if (ky_buffer >= 0 && ky_buffer < rows && kx_buffer >= 0 && kx_buffer < cols)
      {
        x_sum_e = input1.at<float>(ky_buffer, kx_buffer) * (kernel.at<float>(ky, kx));
        x_sum = x_sum + x_sum_e;
      }

    }
  }
  return x_sum;
}
void algorithms::manual_filter_2d(const Mat &input, Mat &output, const Mat &kernel) {
  // TODO put your code here
  int pad;
  if (kernel.rows <= 1){
    pad = (kernel.cols-1)/2;
  } else {
    pad = (kernel.rows -1)/2;
  }
  Mat input1 = Mat::zeros(pad*2,pad*2,CV_32FC1);
  cv::copyMakeBorder(input, input1, pad, pad, pad, pad, BORDER_REPLICATE);

  for (int y = 0; y <input.rows ;y++)
  {
    for (int x = 0 ; x < input.cols; x++)
    {
      output.at<float>(y,x) = sumOfSum(input1,kernel,y,x, input1.rows, input1.cols, pad);
    }
  }
}



//================================================================================
// apply_separable_conv2d()
//--------------------------------------------------------------------------------
// TODO:
//  - apply the convolution for both the y_kernel and the x_kernel and return
//    the resulting Mat. Note: you need to transpose the y_kernel first before
//    applying the convolution. DO NOT use the filter2D function, instead use
//    the manual_filter_2d function.
//
// parameters:
//  - input: single channel 32-bit float matrix
//  - y_kernel: single channel 32-bit float matrix denoting the 1D kernel for
//              the y-dimension
//  - x_kernel: single channel 32-bit float matrix denoting the 1D kernel for
//              the x-dimension
// return: the resulting Mat 32-bit float CV_32F
//================================================================================

Mat algorithms::apply_separable_conv2d(const Mat &input, const Mat &y_kernel, const Mat &x_kernel) {

  Mat y_kernel_transposed, tmp, result;

  cv::transpose(y_kernel, y_kernel_transposed);

  tmp = Mat::zeros(input.rows,input.cols,CV_32FC1);
  result = Mat::zeros(input.rows,input.cols,CV_32FC1);
  manual_filter_2d(input, tmp, y_kernel_transposed);
  manual_filter_2d(tmp, result, x_kernel);

  // TODO put your code here
  return result;
}

//================================================================================
// calc_side_windows()
//--------------------------------------------------------------------------------
// TODO:
//  - Apply all 8 side window filters.
//  - the following side_windows have to be used:
//     1: North West
//     2: South West
//     3: North East
//     4: South East
//     5: West
//     6: East
//     7: North
//     8: South
//
// parameters:
//  - side_windows: vector to store the 8 difference Mat
//  - input_channel: the input image used for the side window filters and the difference
//  - kernel: One of the three 1D kernels (all)
//  - kernel_left: One of the three 1D kernels (left half)
//  - kernel_right: One of the three 1D kernels (right half)
// return: void
//================================================================================

void calc_side_windows(vector<Mat> &side_windows, const Mat &input_channel, const Mat &kernel, const Mat &kernel_left,
                       const Mat &kernel_right) {
  // TODO put your code here

  side_windows.push_back(algorithms::apply_separable_conv2d(input_channel, kernel_left, kernel_left));
  side_windows.push_back(algorithms::apply_separable_conv2d(input_channel, kernel_right, kernel_left));
  side_windows.push_back(algorithms::apply_separable_conv2d(input_channel, kernel_left, kernel_right));
  side_windows.push_back(algorithms::apply_separable_conv2d(input_channel, kernel_right, kernel_right));
  side_windows.push_back(algorithms::apply_separable_conv2d(input_channel, kernel, kernel_left));
  side_windows.push_back(algorithms::apply_separable_conv2d(input_channel, kernel, kernel_right));
  side_windows.push_back(algorithms::apply_separable_conv2d(input_channel, kernel_left, kernel));
  side_windows.push_back(algorithms::apply_separable_conv2d(input_channel, kernel_right, kernel));

}


//================================================================================
// sw_filter()
//--------------------------------------------------------------------------------
// TODO:
//  - Implement the basics of a side window box filter
//  - Split the image into its three channels
//  - For each channel use the calc_side_windows function to calculate the differences
//  - Calculate the absolute values of these differences and push the resulting vector into abs_differences_channel
//  - For each pixel choose the difference with the least absolute difference and apply it to the pixel
//  - You should implement and use the functions calc_side_windows, calc_abs_difference, find_best_fitting_side_window
//
// parameters:
//  - input: the 3 channel input Image
//  - kernel: One of the three 1D kernels (all)
//  - kernel_left: One of the three 1D kernels (left half)
//  - kernel_right: One of the three 1D kernels (right half)
//  - abs_differences_channel: Used for testing should contain the abs differences for each channel
// return: void
//================================================================================

void
algorithms::sw_filter(const Mat input, const Mat &kernel, const Mat &kernel_left, const Mat &kernel_right, Mat &result,
                      vector<vector<Mat>> &abs_differences_channel) {



  std::vector<Mat> channels;
  cv::split(input, channels);


  vector<Mat> side_windows;

  vector<Mat> abs_diff;

  for (int x = 0; x < channels.size(); x++){
    calc_side_windows(side_windows, channels.at(x), kernel, kernel_left, kernel_right);
    calc_abs_difference(side_windows, abs_diff, channels.at(x));
    abs_differences_channel.push_back(abs_diff);
    find_best_fitting_side_window(abs_diff,channels.at(x), side_windows);
    side_windows.clear();
    abs_diff.clear();
  }

  cv::merge(channels,result);
  // TODO put your code here
  imshow("swf", result);
  waitKey(0);
}


//================================================================================
// calc_abs_difference()
//--------------------------------------------------------------------------------
// TODO:
//  - Calculate the absolute difference for each d between d and input and save it
//    in d_abs
//
// parameters:
//  - d: vector with all side windows
//  - d_abs: here all absolute differences should be added
//  - input: the base input for the difference
// return: void
//================================================================================
void algorithms::calc_abs_difference(const vector<Mat> &d, vector<Mat> &d_abs, const Mat &input) {
  // TODO put your code here

  d_abs[0] = cv::abs(input-d[0]);
  d_abs[1] = cv::abs(input-d[1]);
  d_abs[2] = cv::abs(input-d[2]);
  d_abs[3] = cv::abs(input-d[3]);
  d_abs[4] = cv::abs(input-d[4]);
  d_abs[5] = cv::abs(input-d[5]);
  d_abs[6] = cv::abs(input-d[6]);
  d_abs[7] = cv::abs(input-d[7]);

}


//================================================================================
// rgb_to_yuv()
//--------------------------------------------------------------------------------
// TODO:
//  - convert image from BGR to YUV color space
//
// parameters:
//  - bgr_image: The input image in BGR color space
//  - yuv_image: the output image in YUV color space
// return: void
//================================================================================
float checkRange(float value) {
  if(value > 255)
  {
    value = 255;
    return value;
  }
  else if(value < 0)
  {
    value = 0;
    return value;
  }
  else
    return value;
}
void algorithms::bgr_to_yuv(const Mat bgr_image, Mat &yuv_image) {
  bgr_image.copyTo(yuv_image);
  for (int x = 0; x < bgr_image.rows; x++){
    for (int y = 0; y < bgr_image.cols; y++){
      int B = bgr_image.at<cv::Vec3b>(x,y)[0];
      int G = bgr_image.at<cv::Vec3b>(x,y)[1];
      int R = bgr_image.at<cv::Vec3b>(x,y)[2];

      float Y_buffer = 0.299*R + 0.587*G + 0.114*B;
      float U_buffer = (B-Y_buffer) * 0.493;
      float V_buffer = (R-Y_buffer) * 0.877;

      float U_scale = U_buffer + 112;
      float V_scale = (V_buffer/157)*128 + 128;

      uchar Y = cvRound(checkRange(Y_buffer));
      uchar U = cvRound(checkRange(U_scale));
      uchar V = cvRound(checkRange(V_scale));


      yuv_image.at<cv::Vec3b>(x,y)[0] = Y;
      yuv_image.at<cv::Vec3b>(x,y)[1] = U;
      yuv_image.at<cv::Vec3b>(x,y)[2] = V;

    }
  }
  //imshow("test", yuv_image);
  //waitKey(0);
  // TODO put your code here
}

//================================================================================
// visualize_yuv_images()
//--------------------------------------------------------------------------------
// TODO:
//  - visualize Y, U and V channels
//
// parameters:
//  - yuv_image: the YUV input image
//  - u_map: color map to visualize the U channel
//  - v_map: color map to visualize the V channel
//  - yuv_visualized_images: output vector should contain the the Y, U and V channel images
// return: void
//================================================================================

void algorithms::visualize_yuv_channels(const Mat &yuv_image, const std::vector<std::vector<int> > u_color_map,
                                        const std::vector<std::vector<int> > v_color_map,
                                        std::vector<Mat> &yuv_visualized_images) {
  // TODO put your code here

  std::vector<Mat> channels = yuv_visualized_images;
  cv::split(yuv_image, yuv_visualized_images);
  Mat Y_channel = yuv_visualized_images[0];
  Mat U_channel = yuv_visualized_images[1];
  Mat V_channel = yuv_visualized_images[2];

  cv::cvtColor(Y_channel, channels[0], COLOR_GRAY2BGR);
  //cv::cvtColor(U_channel,channels[1], COLOR_GRAY2BGR);
  //cv::cvtColor(V_channel, channels[2], COLOR_GRAY2BGR);
  //imshow("y",BGR_Image);
  //waitKey(0);

  for (int x = 0; x < yuv_image.rows; x++){
    for (int y = 0; y < yuv_image.cols; y++){
      channels[1].at<cv::Vec3b>(x,y)[0] = u_color_map.at(0).at(U_channel.at<uchar>(x,y));
      channels[1].at<cv::Vec3b>(x,y)[1] = u_color_map.at(1).at(U_channel.at<uchar>(x,y));
      channels[1].at<cv::Vec3b>(x,y)[2] = u_color_map.at(2).at(U_channel.at<uchar>(x,y));
      channels[2].at<cv::Vec3b>(x,y)[0] = v_color_map.at(0).at(V_channel.at<uchar>(x,y));
      channels[2].at<cv::Vec3b>(x,y)[1] = v_color_map.at(1).at(V_channel.at<uchar>(x,y));
      channels[2].at<cv::Vec3b>(x,y)[2] = v_color_map.at(2).at(V_channel.at<uchar>(x,y));

    }
  }
  //imshow("y",channels[2]);
  //waitKey(0);
}

//================================================================================
// yuv_to_rgb()
//--------------------------------------------------------------------------------
// TODO:
//  - convert image from YUV to BGR color space
//
// parameters:
//  - yuv_image: The input image in YUV color space
//  - bgr_image: The output image in BGR color space
// return: void
//================================================================================

void algorithms::yuv_to_bgr(Mat yuv_image, Mat &rgb_image)
{
  yuv_image.copyTo(rgb_image);
  for (int x = 0; x < yuv_image.rows; x++)
  {
    for (int y = 0; y < yuv_image.cols; y++)
    {
      float Y = yuv_image.at<cv::Vec3b>(x, y)[0];
      float U = yuv_image.at<cv::Vec3b>(x, y)[1] - 112;
      float V = ((yuv_image.at<cv::Vec3b>(x, y)[2] - 128)*157)/128.;

      float B_buffer = Y + (U/0.492);
      float R_buffer = Y + (V/0.877);
      float G_buffer = 1.704*Y - 0.509*R_buffer - 0.194*B_buffer;

      uchar B = cvRound(checkRange(B_buffer));
      uchar G = cvRound(checkRange(G_buffer));
      uchar R = cvRound(checkRange(R_buffer));

      rgb_image.at<cv::Vec3b>(x,y)[0] = B;
      rgb_image.at<cv::Vec3b>(x,y)[1] = G;
      rgb_image.at<cv::Vec3b>(x,y)[2] = R;

    }
  }
//imshow("test", rgb_image);
  //waitKey(0);
  // TODO put your code here
}


//================================================================================
// find_best_fitting_side_window()
//--------------------------------------------------------------------------------
// TODO:
//  - Find the best fitting side window given the absolute differences
//  - Set the value from the respective filtered images to the output channel
//
// parameters:
//  - d_abs: The absolute differences for each side window
//  - channel: The output channel to change
//  - filtered_imgs: The filtered images for each side window
// return: void
//================================================================================

void algorithms::find_best_fitting_side_window(vector<Mat> &d_abs, Mat &channel, vector<Mat> &filtered_imgs) {
  // TODO put your code here
  for (int x = 0; x<channel.rows;x++){
    for (int y = 0; y<channel.cols;y++){
      vector<float> v = {d_abs[0].at<float>(x,y),d_abs[1].at<float>(x,y),d_abs[2].at<float>(x,y),
                         d_abs[3].at<float>(x,y), d_abs[4].at<float>(x,y), d_abs[5].at<float>(x,y),
                         d_abs[6].at<float>(x,y), d_abs[7].at<float>(x,y)};
      int minIndex = min_element(v.begin(),v.end())-v.begin();
      channel.at<float>(x,y) = filtered_imgs[minIndex].at<float>(x,y);

    }
  }
}

//================================================================================
//================================== BONUS =======================================
//================================================================================

//================================================================================
// bonus()
//--------------------------------------------------------------------------------
// TODO:
//  - Implement the side window bilateral filter
//  - For each channel and kernel use bilateral_filter
//  - Calculate the absolute values of these differences
//  - For each pixel choose the difference with the least absolute difference and apply it to the pixel
//
// parameters:
//  - input: the 3 channel input Image
//  - kernel: One of the three 1D kernels (all)
//  - kernel_left: One of the three 1D kernels (left half)
//  - kernel_right: One of the three 1D kernels (right half)
//  - output: Filtered output image
// return: void
//================================================================================

void algorithms::bonus(Mat &input, const Mat &kernel, const Mat &kernel_left, const Mat &kernel_right, Mat &output) {
  input.copyTo(output);
  vector<Mat> channels;
  cv::split(input, channels);


  // TODO put your code here

}

//================================================================================
// create_bilateral_masks()
//--------------------------------------------------------------------------------
// TODO:
//  - Compute the bilateral kernels according to the assignment sheet
//
// parameters:
//  - kernel: One of the three 1D kernels (all)
//  - kernel_left: One of the three 1D kernels (left half)
//  - kernel_right: One of the three 1D kernels (right half)
//  - kernels: vector in which each kernel should be pushed
// return: void
//================================================================================
void algorithms::create_bilateral_masks(const Mat &kernel, const Mat &kernel_left, const Mat &kernel_right,
                                        vector<Mat> &kernels) {
  // TODO put your code here

}

//================================================================================
// compute_bilateral_weight()
//--------------------------------------------------------------------------------
// TODO:
//  - Compute the weight according to the assignment sheet
//
// parameters:
//  - input:  The single channel input Image
//  - i: Row index of the input image
//  - j: Column index of the input image
//  - k: Row index of the kernel window
//  - l: Column index of the kernel window
//  - sigma_d: standard deviation of spatial gauss distribution
// return: w: The calculated weight
//================================================================================

float algorithms::compute_bilateral_weight(const Mat &input, const int i, const int j, const int k,
                                           const int l, const float sigma_d) {
  const float sigma_r = 0.05;

  // TODO put your code here
}

//================================================================================
// bilateral_filter()
//--------------------------------------------------------------------------------
// TODO:
//  - Implement Bilateral filtering
//  - Create border image - use cv::BORDER_REPLICATE
//  - Iterate over each row and each column of the input image
//  - Use kernel to check whether a pixel is within the current side window
//  - For every pixel use compute_bilateral_weight (needs to be implemented as well) to compute the respective weight
//
// parameters:
//  - input:  The single channel input Image
//  - kernel: One of the 8 side window kernels
//  - output: Filtered image
// return: void
//================================================================================

void algorithms::bilateral_filter(const Mat &input, const Mat &kernel, Mat &output) {

  int k_rows = (kernel.rows - 1) / 2;
  int k_cols = (kernel.cols - 1) / 2;

  float sigma_d = float(k_rows) / 2 + 1;



  // TODO put your code here
}

