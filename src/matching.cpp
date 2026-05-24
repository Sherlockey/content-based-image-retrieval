/*
    Jonathon Davis
    2026-05-24
    Matching functions which, given a database of images and a target image,
   find images in the data with similar content
*/

#include "matching.hpp"
#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>

/*
    Uses a region_dimension square in the middle of the image as a feature
   vector. Uses sum-of-squared-difference as the distance metric

   @param target_path the target image path
   @param database_directory the directory path for the database of images
   @param num_out_images the number of image names to output
   @param region_dimension the dimension of the region observed
*/
std::vector<std::pair<float, std::string>>
baseline(std::string_view target_path, std::string_view database_directory,
         int region_dimension) {
    // open target image
    cv::Mat target;
    std::string target_path_str = std::string(target_path);
    target = cv::imread(target_path_str);

    // test if the read was successful
    if (target.data == NULL) {
        std::cout << "error: unable to read image" << target_path << std::endl;
        exit(1);
    }

    // verify target image is large enough
    if (target.rows < 7 || target.cols < 7) {
        std::cout << "error: image " << target_path << " is less than 7x7"
                  << std::endl;
        exit(1);
    }

    // cache info prior to loop
    int half_dim = region_dimension / 2;
    int target_start_row = target.rows / 2 - half_dim;
    int target_start_col = target.cols / 2 - half_dim;

    std::vector<std::pair<float, std::string>> results;

    // loop through each image in data_base_directory
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(database_directory)) {

        std::string image_path = entry.path().string();
        cv::Mat image = cv::imread(image_path);

        // // test if the read was successful
        if (image.data == NULL) {
            std::cout << "error: unable to read image" << image_path
                      << std::endl;
            exit(1);
        }

        // verify image is large enough
        if (image.rows < 7 || image.cols < 7) {
            std::cout << "error: image " << image_path << " is less than 7x7"
                      << std::endl;
            exit(1);
        }

        int image_start_row = image.rows / 2 - half_dim;
        int image_start_col = image.cols / 2 - half_dim;
        // loop through region and calculate sum-of-squared-difference
        float ssd = 0.0; // sum of squared difference
        for (int i = 0; i < region_dimension; i++) {
            cv::Vec3b* target_ptr = target.ptr<cv::Vec3b>(target_start_row + i);
            cv::Vec3b* image_ptr = image.ptr<cv::Vec3b>(image_start_row + i);
            for (int j = 0; j < region_dimension; j++) {
                for (int k = 0; k < target.channels(); k++) {
                    float difference = target_ptr[target_start_col + j][k] -
                                       image_ptr[image_start_col + j][k];
                    ssd += difference * difference;
                }
            }
        }
        results.emplace_back(ssd, image_path);
    }
    std::sort(results.begin(), results.end());
    return results;
}

/*
   TODO helper
*/
cv::Mat compute_histogram(cv::Mat& image, int buckets) {
    int sizes[] = {buckets, buckets, buckets};
    cv::Mat image_hist(3, sizes, CV_32F, cv::Scalar(0));
    for (int i = 0; i < image.rows; i++) {
        cv::Vec3b* image_ptr = image.ptr<cv::Vec3b>(i);
        for (int j = 0; j < image.cols; j++) {
            int b = image_ptr[j][0] * buckets / 256;
            int g = image_ptr[j][1] * buckets / 256;
            int r = image_ptr[j][2] * buckets / 256;
            image_hist.at<float>(b, g, r) += 1.0f;
        }
    }
    image_hist /= (image.rows * image.cols);
    return image_hist;
}

/*
   TODO helper
*/
float histogram_intersection_distance(const cv::Mat& a, const cv::Mat& b,
                                      int buckets) {
    float intersection = 0.0;
    for (int i = 0; i < buckets; i++) {
        for (int j = 0; j < buckets; j++) {
            for (int k = 0; k < buckets; k++) {
                intersection +=
                    std::min(a.at<float>(i, j, k), b.at<float>(i, j, k));
            }
        }
    }
    return 1.0f - intersection;
}

/*
   TODO
*/
std::vector<std::pair<float, std::string>>
histogram(std::string_view target_path, std::string_view database_directory,
          int buckets) {
    if (buckets <= 0) {
        std::cout << "error: cannot use histogram with buckets <= 0"
                  << std::endl;
        exit(1);
    }

    cv::Mat target;
    std::string target_path_str = std::string(target_path);
    target = cv::imread(target_path_str);

    // test if the read was successful
    if (target.data == NULL) {
        std::cout << "error: unable to read image" << target_path << std::endl;
        exit(1);
    }

    // make target_histogram
    cv::Mat target_histogram = compute_histogram(target, buckets);

    std::vector<std::pair<float, std::string>> results;

    // loop through each image in data_base_directory
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(database_directory)) {

        std::string image_path = entry.path().string();
        cv::Mat image = cv::imread(image_path);

        // // test if the read was successful
        if (image.data == NULL) {
            std::cout << "error: unable to read image" << image_path
                      << std::endl;
            exit(1);
        }

        // make image_histogram and compute/store distance
        cv::Mat image_histogram = compute_histogram(image, buckets);
        float distance = histogram_intersection_distance(
            target_histogram, image_histogram, buckets);
        results.emplace_back(distance, image_path);
    }
    std::sort(results.begin(), results.end());
    return results;
}

void multi_histogram(std::string_view target_path,
                     std::string_view database_directory,
                     std::string_view feature_method,
                     std::string_view distance_metric, int num_out_images) {}

void texture_color(std::string_view target_path,
                   std::string_view database_directory,
                   std::string_view feature_method,
                   std::string_view distance_metric, int num_out_images) {}

void deep_network_embeddings(std::string_view target_path,
                             std::string_view database_directory,
                             std::string_view feature_method,
                             std::string_view distance_metric,
                             int num_out_images) {}
