#pragma once

#include <opencv2/opencv.hpp>
#include <string_view>

std::vector<std::pair<double, std::string>>
baseline(std::string_view target_path, std::string_view database_directory,
         int num_out_images, int region_dimension = 7);

void histogram(std::string_view target_path,
               std::string_view database_directory,
               std::string_view feature_method,
               std::string_view distance_metric, int num_out_images);

void multi_histogram(std::string_view target_path,
                     std::string_view database_directory,
                     std::string_view feature_method,
                     std::string_view distance_metric, int num_out_images);

void texture_color(std::string_view target_path,
                   std::string_view database_directory,
                   std::string_view feature_method,
                   std::string_view distance_metric, int num_out_images);

void deep_network_embeddings(std::string_view target_path,
                             std::string_view database_directory,
                             std::string_view feature_method,
                             std::string_view distance_metric,
                             int num_out_images);
