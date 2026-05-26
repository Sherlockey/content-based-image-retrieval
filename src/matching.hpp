/*
    Jonathon Davis
    2026-05-24
    Matching functions which, given a database of images and a target image,
   find images in the data with similar content
*/

#pragma once

#include <opencv2/opencv.hpp>
#include <string_view>

enum class DistanceMetric {
    SumOfSquaredDistance,
    HistogramIntersection,
    CosineDistance
};

// helper to parse from string-view
DistanceMetric parse_distance_metric(std::string_view name);

// helper to print metric name
const char* distance_metric_name(DistanceMetric m);

// prototypes
std::vector<std::pair<float, std::string>>
baseline(std::string_view target_path, std::string_view database_directory,
         int region_dimension = 7);

std::vector<std::pair<float, std::string>>
histogram(std::string_view target_path, std::string_view database_directory,
          int buckets = 16,
          DistanceMetric metric = DistanceMetric::HistogramIntersection);

std::vector<std::pair<float, std::string>>
multi_histogram(std::string_view target_path,
                std::string_view database_directory, int buckets = 16,
                float edge_offset = 0.25, float uncropped_weight = 0.5f,
                DistanceMetric metric = DistanceMetric::HistogramIntersection);

std::vector<std::pair<float, std::string>>
texture_color(std::string_view target_path, std::string_view database_directory,
              int buckets = 16, float color_weight = 0.5f,
              DistanceMetric metric = DistanceMetric::HistogramIntersection);

std::vector<std::pair<float, std::string>> deep_network_embeddings(
    const char* target, const char* dnn_embeddings,
    DistanceMetric metric = DistanceMetric::SumOfSquaredDistance);

std::vector<std::pair<float, std::string>>
hsv_histogram(std::string_view target_path, std::string_view database_directory,
              int buckets = 16,
              DistanceMetric metric = DistanceMetric::CosineDistance);

std::vector<std::pair<float, std::string>>
orientation_color(std::string_view target_path,
                  std::string_view database_directory, int color_buckets = 16,
                  int orientation_buckets = 9, float color_weight = 0.5f,
                  DistanceMetric metric = DistanceMetric::CosineDistance);

std::vector<std::pair<float, std::string>>
combined_features(std::string_view target_path,
                  std::string_view database_directory, int color_buckets = 16,
                  int orientation_buckets = 9, float edge_offset = 0.25f);
