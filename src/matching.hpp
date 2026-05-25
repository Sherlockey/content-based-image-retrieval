#pragma once

#include <opencv2/opencv.hpp>
#include <string_view>

// prototypes
std::vector<std::pair<float, std::string>>
baseline(std::string_view target_path, std::string_view database_directory,
         int region_dimension = 7);

std::vector<std::pair<float, std::string>>
histogram(std::string_view target_path, std::string_view database_directory,
          int buckets = 16);

std::vector<std::pair<float, std::string>>
multi_histogram(std::string_view target_path,
                std::string_view database_directory, int buckets = 16,
                float edge_offset = 0.25, float uncropped_weight = 0.5f);

std::vector<std::pair<float, std::string>>
texture_color(std::string_view target_path, std::string_view database_directory,
              int buckets = 16, float color_weight = 0.5f);

std::vector<std::pair<float, std::string>>
deep_network_embeddings(const char* target, const char* dnn_embeddings);

std::vector<std::pair<float, std::string>>
hsv_histogram(std::string_view target_path, std::string_view database_directory,
              int buckets = 16);

std::vector<std::pair<float, std::string>>
orientation_color(std::string_view target_path,
                  std::string_view database_directory, int color_buckets = 16,
                  int orientation_buckets = 9, float color_weight = 0.5f);
