/*
    Jonathon Davis
    2026-05-24
    Matching functions which, given a database of images and a target image,
   find images in the data with similar content
*/

#include "matching.hpp"
#include <filesystem>
#include <iostream>

/*
    Uses a 7x7 square in the middle of the image as a feature vector. Uses
   sum-of-squared-difference as the distance metric

   @param target_path the target image path
   @param database_directory the directory path for the database of images
   @param num_out_images the number of image names to output
   @param region_dimension the dimension of the region observed
*/
std::vector<std::pair<double, std::string>>
baseline(std::string_view target_path, std::string_view database_directory,
         int num_out_images, int region_dimension) {
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

    std::vector<std::pair<double, std::string>> results;

    // loop through each image in data_base_directory
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(database_directory)) {

        cv::Mat image;
        std::string image_path = entry.path().string();
        image = cv::imread(image_path);

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
        double ssd = 0.0; // sum of squared difference
        for (int i = 0; i < region_dimension; i++) {
            cv::Vec3b* target_ptr = target.ptr<cv::Vec3b>(target_start_row + i);
            cv::Vec3b* image_ptr = image.ptr<cv::Vec3b>(image_start_row + i);
            for (int j = 0; j < region_dimension; j++) {
                for (int k = 0; k < target.channels(); k++) {
                    double difference = target_ptr[target_start_col + j][k] -
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

void histogram(std::string_view target_path,
               std::string_view database_directory,
               std::string_view feature_method,
               std::string_view distance_metric, int num_out_images) {}

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
