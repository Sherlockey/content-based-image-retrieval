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
    Compares a target image to a database of images.
    Uses a region_dimension square in the middle of the image as a feature
   vector.
    Uses sum-of-squared-difference as the distance metric.

    @param target_path the target image path
    @param database_directory the directory path for the database of images
    @param num_out_images the number of image names to output
    @param region_dimension the dimension of the region observed
    @return the vector containing pairings between a float value and the string
    pathname
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
    Helper, computes and returns a 3D color histogram.

    @param image the image to perform the operation on
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @return the computed 3D color histogram
*/
cv::Mat compute_histogram(cv::Mat& image, int buckets) {
    int sizes[] = {buckets, buckets, buckets};
    cv::Mat histogram(3, sizes, CV_32F, cv::Scalar(0));

    for (int i = 0; i < image.rows; i++) {
        cv::Vec3b* image_ptr = image.ptr<cv::Vec3b>(i);
        for (int j = 0; j < image.cols; j++) {
            // calculate then store running bgr counts in histogram
            int b = image_ptr[j][0] * buckets / 256;
            int g = image_ptr[j][1] * buckets / 256;
            int r = image_ptr[j][2] * buckets / 256;
            histogram.at<float>(b, g, r) += 1.0f;
        }
    }
    histogram /= (image.rows * image.cols);
    return histogram;
}

/*
    Helper, calculates histogram intersection distance between two histograms.

    @param a histogram a
    @param b histogram b
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @return the computed histogram intersection distance between a and b
*/
float histogram_intersection_distance(const cv::Mat& a, const cv::Mat& b,
                                      int buckets) {
    float intersection = 0.0;
    for (int i = 0; i < buckets; i++) {
        for (int j = 0; j < buckets; j++) {
            for (int k = 0; k < buckets; k++) {
                // sum mins at all i, j, k
                intersection +=
                    std::min(a.at<float>(i, j, k), b.at<float>(i, j, k));
            }
        }
    }
    return 1.0f - intersection; // 1 - intersection gives distance
}

/*
    Compares a target image to a database of images.
    Uses a 3D color histogram as the feature vector.
    Uses histogram intersection as the distance metric.

    @param target_path the target image path
    @param database_directory the directory path for the database of images
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @return the vector containing pairings between a float value and the string
    pathname
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

/*
    Compares a target image to a database of images in two ways. First the whole
   images, second a region of the center of the images. Uses a 3D color
   histogram as the feature vector. Uses histogram intersection as the distance
   metric.

    @param target_path the target image path
    @param database_directory the directory path for the database of images
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @param x the starting x of the center region
    @param y the starting y of the center region
    @param width the width of the center region
    @param height the height of the center region
    @param uncropped_weight the weight given to the uncropped region (between
   0.0f and 1.0f), the cropped_weight will be (1.0f - uncropped_weight)
    @return the vector containing pairings between a float value and the string
    pathname
*/
std::vector<std::pair<float, std::string>>
multi_histogram(std::string_view target_path,
                std::string_view database_directory, int buckets, int x, int y,
                int width, int height, float uncropped_weight) {

    if (uncropped_weight < 0.0f || uncropped_weight > 1.0f) {
        std::cout << "error: cannot run multi_histogram with a "
                     "full_image_weight of < 0.0f or > 1.0f"
                  << std::endl;
        exit(1);
    }

    cv::Mat target;
    target = cv::imread(std::string(target_path));
    // test if the read was successful
    if (target.data == NULL) {
        std::cout << "error: unable to read image" << target_path << std::endl;
        exit(1);
    }

    // validate ROI fits within the target
    if (x < 0 || y < 0 || x + width > target.cols || y + height > target.rows) {
        std::cout
            << "error: region of interest does not fit within target image"
            << std::endl;
        exit(1);
    }
    cv::Rect roi(x, y, width, height);

    // compute both target histograms (uncropped and cropped)
    cv::Mat target_uncropped_histogram = compute_histogram(target, buckets);
    cv::Mat target_cropped = target(roi);
    cv::Mat target_cropped_histogram =
        compute_histogram(target_cropped, buckets);

    float cropped_weight = 1.0f - uncropped_weight;
    std::vector<std::pair<float, std::string>> results;

    // loop through each image in data_base_directory
    for (const auto& entry :
         std::filesystem::directory_iterator(database_directory)) {

        std::string image_path = entry.path().string();
        cv::Mat image = cv::imread(image_path);
        if (image.data == NULL) {
            std::cout << "error: unable to read image" << image_path
                      << std::endl;
            exit(1);
        }

        // compute both histograms for this image
        cv::Mat image_uncropped_histogram = compute_histogram(image, buckets);
        cv::Mat image_cropped = image(roi);
        cv::Mat image_cropped_histogram =
            compute_histogram(image_cropped, buckets);

        // compute both distances
        float uncropped_distance = histogram_intersection_distance(
            target_uncropped_histogram, image_uncropped_histogram, buckets);
        float cropped_distance = histogram_intersection_distance(
            target_cropped_histogram, image_cropped_histogram, buckets);

        // weighted average
        float combined = uncropped_weight * uncropped_distance +
                         cropped_weight * cropped_distance;
        results.emplace_back(combined, image_path);
    }
    std::sort(results.begin(), results.end());
    return results;
}

void texture_color(std::string_view target_path,
                   std::string_view database_directory,
                   std::string_view feature_method,
                   std::string_view distance_metric, int num_out_images) {}

void deep_network_embeddings(std::string_view target_path,
                             std::string_view database_directory,
                             std::string_view feature_method,
                             std::string_view distance_metric,
                             int num_out_images) {}
