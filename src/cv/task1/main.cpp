#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "algorithms.h"

#define FULL_VERSION 1

#define RST "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST
#define FYEL(x) KYEL x RST
#define FBLU(x) KBLU x RST
#define FMAG(x) KMAG x RST
#define FCYN(x) KCYN x RST
#define FWHT(x) KWHT x RST

#define BOLD(x) "\x1B[1m" x RST
#define UNDL(x) "\x1B[4m" x RST

using namespace std;
using namespace cv;

int loadConfig(rapidjson::Document &config, const char *config_path) {
  FILE *fp = fopen(config_path, "r");
  if (!fp) {
    cout << BOLD(FRED("[ERROR]")) << " Reading File " << config_path << " failed\n" << endl;
    return -1;
  }
  char readBuffer[65536];
  rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
  config.ParseStream(is);
  assert(config.IsObject());
  return 0;
}

void createDir(const char *path) {
#if defined(_WIN32)
  _mkdir(path);
#else
  mkdir(path, 0777);
#endif
}

vector<string> getDataSelections(rapidjson::Document &config, string data_selection) {
  const rapidjson::Value &a = config[data_selection.c_str()];
  vector<string> members;
  for (rapidjson::SizeType i = 0; i < a.Size(); i++) {
    if (a[i].IsString()) {
      string data_selected = a[i].GetString();
      members.push_back(data_selected);
    }
  }
  return members;
}

vector<string>
getConfigFilenames(rapidjson::Document &config, int number_width, bool zero_filled, const string out_filename_array,
                   const string out_full_path, const string out_filetype) {
  const rapidjson::Value &a = config[out_filename_array.c_str()];
  vector<string> members;
  for (rapidjson::SizeType i = 0; i < a.Size(); i++) {
    if (a[i].IsString()) {
      string out_filename = a[i].GetString();
      string digit_string = to_string(i + 1);
      string zero_digit_string = zero_filled ? string(number_width - digit_string.length(), '0') + digit_string
                                             : digit_string;
      string member = out_filename + zero_digit_string;
      members.emplace_back(out_full_path + zero_digit_string + "_" + out_filename + out_filetype);
//      cout << member << " = " << members.back() << endl;
    }
  }
  return members;
}

std::string replaceString(
    std::string s,
    const std::string &toReplace,
    const std::string &replaceWith) {
  std::size_t pos = s.find(toReplace);
  if (pos == std::string::npos) return s;
  return s.replace(pos, toReplace.length(), replaceWith);
}

void run(Mat image, vector<string> out_filenames, string out_filetype, std::size_t no_iterations, std::size_t radius,
         double sigma) {

  Mat k, k_L, k_R, k_gauss_2d, k_gauss_1d, img_save, yuv_img, result, iter, iter_gaus_2d, iter_gaus_1d, iter_bil;
  int out_filename_count_snapshot, out_filename_count = 0;
  clock_t start_swf, end_swf, start_gaus_2d, end_gaus_2d, start_gaus_1d, end_gaus_1d, start_bil, end_bil;

  //====================================================================================================================
  // Initialze the kernels and store them
  //====================================================================================================================
  algorithms::initializer_kernel_matrix(radius, sigma, k_L, k_R, k, k_gauss_2d, k_gauss_1d);
  k.convertTo(img_save, CV_8UC3, 255.f);
  if (!img_save.empty())
    imwrite(out_filenames[out_filename_count], img_save);
  out_filename_count++;
  k_L.convertTo(img_save, CV_8UC3, 255.f);
  if (!img_save.empty())
    imwrite(out_filenames[out_filename_count], img_save);
  out_filename_count++;
  k_R.convertTo(img_save, CV_8UC3, 255.f);
  if (!img_save.empty())
    imwrite(out_filenames[out_filename_count], img_save);
  out_filename_count++;
  k_gauss_2d.convertTo(img_save, CV_8UC3, 255.f);
  if (!img_save.empty())
    imwrite(out_filenames[out_filename_count], img_save);
  out_filename_count++;
  k_gauss_1d.convertTo(img_save, CV_8UC3, 255.f);
  if (!img_save.empty())
    imwrite(out_filenames[out_filename_count], img_save);
  out_filename_count++;


  //====================================================================================================================
  // apply the swf to the image
  //====================================================================================================================
  out_filename_count_snapshot = out_filename_count;
  image.convertTo(iter, CV_32F, 1 / 255.0);
  vector<vector<Mat>> d;
  start_swf = clock();
  for (size_t i = 0; i < no_iterations; i++) {
    out_filename_count = out_filename_count_snapshot;
    algorithms::sw_filter(iter, k, k_L, k_R, result, d);

    // store iteration results
    if (i <= 4 || !(i % 5)) {
      result.convertTo(img_save, CV_8UC3, 255.f);
      if (!img_save.empty())
        imwrite(replaceString(out_filenames[out_filename_count], out_filetype, "_it_" + to_string(i) + out_filetype),
                img_save);
      out_filename_count++;
      if (d.size() != 3) {
        cout << BOLD(FGRN("[INFO]"))
             << " You have pushed a wrong number of channels to abs_differences_channel (Should be 3)" << endl;
      } else {
        for (int d_i = 0; d_i < min({d.at(0).size(), d.at(1).size(), d.at(2).size()}); d_i++) {
          vector<Mat> m = {d.at(0).at(d_i), d.at(1).at(d_i), d.at(2).at(d_i)};
          merge(m, img_save);
          img_save.convertTo(img_save, CV_8UC3, 255.f);
          if (!img_save.empty())
            imwrite(replaceString(out_filenames[out_filename_count], out_filetype,
                                  "_it_" + to_string(i) + out_filetype), img_save);
        }
      }
    }
    d.clear();
    iter = result;
  }
  end_swf = clock();
  out_filename_count = out_filename_count_snapshot + 2;
  result.convertTo(result, CV_8U, 255.f);
  if (!result.empty())
    imwrite(out_filenames[out_filename_count], result);
  out_filename_count++;

  //====================================================================================================================
  // Gaussian blur with a 2D kernel
  //====================================================================================================================
  image.convertTo(iter_gaus_2d, CV_32FC3, 1 / 255.0);
  vector<Mat> channels(3);
  cv::split(iter_gaus_2d, channels);
  start_gaus_2d = clock();
  for (size_t i = 0; i < no_iterations; i++) {
    algorithms::manual_filter_2d(channels.at(0), channels.at(0), k_gauss_2d);
    algorithms::manual_filter_2d(channels.at(1), channels.at(1), k_gauss_2d);
    algorithms::manual_filter_2d(channels.at(2), channels.at(2), k_gauss_2d);

    // store iteration results
    if (i <= 4 || !(i % 5)) {
      cv::merge(channels, iter_gaus_2d);
      iter_gaus_2d.convertTo(img_save, CV_8U, 255.f);
      if (!img_save.empty())
        imwrite(replaceString(out_filenames[out_filename_count], out_filetype, "_it_" + to_string(i) + out_filetype),
                img_save);
    }
  }
  end_gaus_2d = clock();
  out_filename_count++;
  cv::merge(channels, iter_gaus_2d);
  iter_gaus_2d.convertTo(img_save, CV_8U, 255.f);
  if (!img_save.empty())
    imwrite(out_filenames[out_filename_count], img_save);
  out_filename_count++;


  //====================================================================================================================
  // Gaussian blur with a separated 1D kernel
  //====================================================================================================================
  image.convertTo(iter_gaus_1d, CV_32FC3, 1 / 255.0);
  cv::split(iter_gaus_1d, channels);
  start_gaus_1d = clock();
  for (size_t i = 0; i < no_iterations; i++) {
    channels.at(0) = algorithms::apply_separable_conv2d(channels.at(0), k_gauss_1d, k_gauss_1d);
    channels.at(1) = algorithms::apply_separable_conv2d(channels.at(1), k_gauss_1d, k_gauss_1d);
    channels.at(2) = algorithms::apply_separable_conv2d(channels.at(2), k_gauss_1d, k_gauss_1d);

    // store iteration results
    if (i <= 4 || !(i % 5)) {
      cv::merge(channels, iter_gaus_1d);
      iter_gaus_1d.convertTo(img_save, CV_8U, 255.f);
      if (!img_save.empty())
        imwrite(replaceString(out_filenames[out_filename_count], out_filetype, "_it_" + to_string(i) + out_filetype),
                img_save);
    }
  }
  end_gaus_1d = clock();
  out_filename_count++;
  cv::merge(channels, iter_gaus_1d);
  iter_gaus_1d.convertTo(img_save, CV_8U, 255.f);
  if (!img_save.empty())
    imwrite(out_filenames[out_filename_count], img_save);
  out_filename_count++;


  //====================================================================================================================
  // BGR -> YUV
  //====================================================================================================================
  Mat img_yuv = Mat::zeros(image.rows, image.cols, CV_8UC3);
  algorithms::bgr_to_yuv(image, img_yuv);
  if (!img_yuv.empty())
    imwrite(out_filenames[out_filename_count], img_yuv);
  out_filename_count++;
//  compare to opencv implementation
//  img_save = Mat::zeros(image.rows, image.cols, CV_8UC3);
//  cvtColor(image, img_save, COLOR_BGR2YUV);
//  imwrite(out_filenames[out_filename_count++], img_save);

  //====================================================================================================================
  // Visualize YUV channels
  //====================================================================================================================
  Mat y = Mat::zeros(image.rows, image.cols, CV_8UC3);
  Mat u = Mat::zeros(image.rows, image.cols, CV_8UC3);
  Mat v = Mat::zeros(image.rows, image.cols, CV_8UC3);
  std::vector<Mat> visualized_yuv_images{y, u, v};
  std::vector<std::vector<int>> u_color_map;
  std::vector<std::vector<int>> v_color_map;
  std::vector<int> u_r_map, u_g_map, u_b_map;
  std::vector<int> v_r_map, v_g_map, v_b_map;

  for (int i = 0; i < 256; i++) {
    u_r_map.push_back(0);
    u_g_map.push_back(255 - i);
    u_b_map.push_back(i);

    v_r_map.push_back(i);
    v_g_map.push_back(255 - i);
    v_b_map.push_back(0);
  }

  u_color_map.push_back(u_b_map);
  u_color_map.push_back(u_g_map);
  u_color_map.push_back(u_r_map);

  v_color_map.push_back(v_b_map);
  v_color_map.push_back(v_g_map);
  v_color_map.push_back(v_r_map);

  algorithms::visualize_yuv_channels(img_yuv, u_color_map, v_color_map, visualized_yuv_images);

  if (!visualized_yuv_images.empty() && !visualized_yuv_images[0].empty())
    imwrite(replaceString(out_filenames[out_filename_count], out_filetype,
                          "_y" + out_filetype), visualized_yuv_images[0]);
  if (!visualized_yuv_images.empty() && !visualized_yuv_images[1].empty())
    imwrite(replaceString(out_filenames[out_filename_count], out_filetype,
                          "_u" + out_filetype), visualized_yuv_images[1]);
  if (!visualized_yuv_images.empty() && !visualized_yuv_images[2].empty())
    imwrite(replaceString(out_filenames[out_filename_count], out_filetype,
                          "_v" + out_filetype), visualized_yuv_images[2]);
  out_filename_count++;

  //====================================================================================================================
  // YUV -> BGR
  //====================================================================================================================
  img_save = Mat::zeros(image.rows, image.cols, CV_8UC3);
  algorithms::yuv_to_bgr(img_yuv, img_save);
  if (!img_save.empty())
    imwrite(out_filenames[out_filename_count], img_save);
  out_filename_count++;


  //====================================================================================================================
  // apply bilateral filtering with sw
  //====================================================================================================================

  out_filename_count_snapshot = out_filename_count;
  image.convertTo(iter_bil, CV_32F, 1 / 255.0);
  Mat result_bonus;

  start_bil = clock();
  for (size_t i = 0; i < no_iterations; i++) {
    out_filename_count = out_filename_count_snapshot;
    algorithms::bonus(iter_bil, k, k_L, k_R, result_bonus);

    // store iteration results
    if (i <= 4 || !(i % 5)) {
      result_bonus.convertTo(img_save, CV_8UC3, 255.f);
      if (!img_save.empty())
        imwrite(replaceString(out_filenames[out_filename_count], out_filetype, "_it_" + to_string(i) + out_filetype),
                img_save);
      out_filename_count++;

    }
    iter_bil = result_bonus;
  }
  end_bil = clock();
  out_filename_count = out_filename_count_snapshot + 1;
  result_bonus.convertTo(result_bonus, CV_8U, 255.f);
  if (!result_bonus.empty())
    imwrite(out_filenames[out_filename_count], result_bonus);
  out_filename_count++;

  /*
  Mat in_bil, out_bil;
  image.convertTo(in_bil, CV_32F, 1 / 255.0);
  for(int i = 0; i < 5; i++)
  {
      cv::bilateralFilter(in_bil, out_bil, 3, 0.05, 2.0, cv::BORDER_REPLICATE);
      out_bil.copyTo(in_bil);
  }

  cv::imshow("bil", out_bil);
  cv::waitKey();
  */

  //====================================================================================================================
  // Time comparison
  //====================================================================================================================
  cout << "____________________________________________________________" << endl;
  cout << "|             |   Gauss 1D   |   Gauss 2D   |     SWF      |" << endl;
  cout << "|-------------|--------------|--------------|---------------" << endl;
  printf("|  Zeit [sec] |%14f|%14f|%14f|\n", double(end_gaus_1d - start_gaus_1d) / (CLOCKS_PER_SEC),
         double(end_gaus_2d - start_gaus_2d) / (CLOCKS_PER_SEC), double(end_swf - start_swf) / (CLOCKS_PER_SEC));
  cout << "------------------------------------------------------------" << endl;
}

//======================================================================================================================
// main()
//======================================================================================================================
int main(int argc, char *argv[]) {
  printf(BOLD(FGRN("[INFO ]")));
  printf(" CV/task1 framework version 1.0\n");  // DO NOT REMOVE THIS LINE!

  bool load_default_config = argc == 1;
  bool load_argv_config = argc == 2;

  // check console arguments
  if (load_default_config) {
    cout << BOLD(FGRN("[INFO]")) << " No Testcase selected - using default Testcase (=0)\n" << endl;
  } else if (!load_argv_config) {
    cout << BOLD(FRED("[ERROR]")) << " Usage: ./cvtask1 <TC-NO. (0-1)>\n" << endl;
    return -1;
  }

  try {
    // load config
    rapidjson::Document config;
    int res = loadConfig(config, "config.json"); //load std config file!
    if (res != 0)
      return -1;

    // input parameters
    vector<string> data_selections = getDataSelections(config, string("data_selected"));
    if (load_argv_config && atoi(argv[1]) >= data_selections.size()) {
      cout << BOLD(FRED("[ERROR]")) << " Comandline argument (= " << atoi(argv[1])
           << ") is higher than number of Testcases (= " << data_selections.size() - 1 << ")\n" << endl;
      return -1;
    }
    string data_selected = data_selections.at(load_default_config ? 0 : atoi(argv[1]));
    string data_path = config["data_path"].GetString();
    string out_directory = config["out_directory"].GetString();
    string out_full_path = out_directory + data_selected;
    string out_filetype = config["out_filetype"].GetString();

    int out_filename_number_width = config["out_filename_number_width"].GetInt();
    bool out_filename_number_zero_filled = config["out_filename_number_zero_filled"].GetBool();
    vector<string> out_filenames = getConfigFilenames(config, out_filename_number_width,
                                                      out_filename_number_zero_filled, string("out_filenames"),
                                                      out_full_path, out_filetype);

    // combine needed strings
    string data_full_path = data_path + data_selected;
    string data_config_full_path = data_full_path + "config.json"; //specific config file for the TC
    cout << BOLD(FGRN("[INFO]")) << " Data path: " << data_full_path << endl;
    cout << BOLD(FGRN("[INFO]")) << " Data config path: " << data_config_full_path << endl;
    cout << BOLD(FGRN("[INFO]")) << " Output path: " << out_full_path << endl;

    // load data config
    rapidjson::Document config_data;
    res = loadConfig(config_data, data_config_full_path.c_str());
    if (res != 0)
      return -1;

    // load data config content
    string image_path = config_data["image"].GetString();
    size_t no_iterations = config_data["iterations"].GetUint();
    size_t radius = config_data["radius"].GetUint();
    double sigma = config_data["sigma"].GetDouble();

    string image_full_path = data_full_path + image_path;
    cout << BOLD(FGRN("[INFO]")) << " Image path: " << image_full_path << endl;

    // create output dirs
    createDir(out_directory.c_str());
    createDir(out_full_path.c_str());

    // load input image
    Mat img_BGR = imread(image_full_path);
    // check if image was loaded
    if (!img_BGR.data) {
      cout << BOLD(FRED("[ERROR]")) << " Could not load image (" << image_full_path << ")" << endl;
      return -1;
    }

    run(img_BGR, out_filenames, out_filetype, no_iterations, radius, sigma);
  }
  catch (const exception &ex) {
    cout << ex.what() << endl;
    cout << BOLD(FRED("[ERROR]")) << " Program exited with errors!" << endl;
    return -1;
  }
  cout << BOLD(FGRN("[INFO ]")) << " Program exited normally!" << endl;
  return 0;
}
