/*
    Jonathon Davis
    2026-05-24
    Matching functions which, given a database of images and a target image,
   find images in the data with similar content
*/

#include "matching.hpp"
#include "csv_util.h"
#include "filter.h"
#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>

/*
    Helper, to parse the passed distance metric into an enum
*/
DistanceMetric parse_distance_metric(std::string_view name) {
    if (name == "sum-of-squared-distance")
        return DistanceMetric::SumOfSquaredDistance;
    if (name == "histogram-intersection")
        return DistanceMetric::HistogramIntersection;
    if (name == "cosine-distance")
        return DistanceMetric::CosineDistance;
    // unreachable if main validated the arg, but fallback anyways
    return DistanceMetric::SumOfSquaredDistance;
}

/*
    Helper, to get name from enum
*/
const char* distance_metric_name(DistanceMetric m) {
    switch (m) {
    case DistanceMetric::SumOfSquaredDistance:
        return "sum-of-squared-distance";
    case DistanceMetric::HistogramIntersection:
        return "histogram-intersection";
    case DistanceMetric::CosineDistance:
        return "cosine-distance";
    }
    return "unknown";
}

/*
    Helper, if requested metric isn't supported by this feature function, print
   a warning and return the default. Otherwise return the requested metric.
*/
DistanceMetric resolve_metric(DistanceMetric requested,
                              const std::vector<DistanceMetric>& supported,
                              DistanceMetric default_metric,
                              const char* function_name) {
    for (DistanceMetric m : supported) {
        if (m == requested)
            return requested;
    }
    std::cout << "warning: " << function_name << " does not support "
              << distance_metric_name(requested) << ". Falling back to "
              << distance_metric_name(default_metric) << std::endl;
    return default_metric;
}

/*
    Helper, SSD between two 3D histograms.
*/
float ssd_3d(cv::Mat& a, cv::Mat& b, int buckets) {
    double ssd = 0.0;
    for (int i = 0; i < buckets; i++) {
        for (int j = 0; j < buckets; j++) {
            for (int k = 0; k < buckets; k++) {
                double diff = a.at<float>(i, j, k) - b.at<float>(i, j, k);
                ssd += diff * diff;
            }
        }
    }
    return static_cast<float>(ssd);
}

/*
    Helper, SSD between two 1D histograms.
*/
float ssd_1d(cv::Mat& a, cv::Mat& b, int buckets) {
    double ssd = 0.0;
    for (int i = 0; i < buckets; i++) {
        double diff = a.at<float>(i) - b.at<float>(i);
        ssd += diff * diff;
    }
    return static_cast<float>(ssd);
}

/*
    Helper, histogram intersection distance for 1D histograms.
*/
float histogram_intersection_distance_1d(cv::Mat& a, cv::Mat& b, int buckets) {
    float intersection = 0.0f;
    for (int i = 0; i < buckets; i++) {
        intersection += std::min(a.at<float>(i), b.at<float>(i));
    }
    return 1.0f - intersection;
}

/*
    Helper, cosine distance between two 3D histograms.

    @param a histogram a
    @param b histogram b
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @return cosine distance
 */
float cosine_distance_3d(cv::Mat& a, cv::Mat& b, int buckets) {
    double dot = 0.0;    // dot product
    double norm_a = 0.0; // sum of squares of a's components OR ||a||^2
    double norm_b = 0.0; // sum of squares for b's components
    for (int i = 0; i < buckets; i++) {
        for (int j = 0; j < buckets; j++) {
            for (int k = 0; k < buckets; k++) {
                float av = a.at<float>(i, j, k);
                float bv = b.at<float>(i, j, k);
                dot += av * bv;
                norm_a += av * av;
                norm_b += bv * bv;
            }
        }
    }
    if (norm_a == 0 || norm_b == 0) {
        // undefined / max distance. return completely dissimilar in this
        // case
        return 1.0f;
    }
    double similarity = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    return static_cast<float>(1.0 - similarity); // 1 - similarity, aka distance
}

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
    cv::Mat target = cv::imread(std::string(target_path));

    // test if the read was successful
    if (target.data == NULL) {
        std::cout << "error: unable to read image " << target_path << std::endl;
        exit(1);
    }

    // verify target image is large enough
    if (target.rows < region_dimension || target.cols < region_dimension) {
        std::cout << "error: image " << target_path << " is less than "
                  << region_dimension << "x" << region_dimension << std::endl;
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
            std::cout << "error: unable to read image " << image_path
                      << std::endl;
            exit(1);
        }

        // verify image is large enough
        if (image.rows < region_dimension || image.cols < region_dimension) {
            std::cout << "error: image " << image_path << " is less than "
                      << region_dimension << "x" << region_dimension
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
float histogram_intersection_distance(cv::Mat& a, cv::Mat& b, int buckets) {
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
    Uses a RGB as the feature vector for the histogram.

    @param target_path the target image path
    @param database_directory the directory path for the database of images
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @param DistanceMetric the distance metric to use when comparing features
    @return the vector containing pairings between a float value and the string
    pathname
*/
std::vector<std::pair<float, std::string>>
histogram(std::string_view target_path, std::string_view database_directory,
          int buckets, DistanceMetric metric) {
    if (buckets <= 0) {
        std::cout << "error: cannot use histogram with buckets <= 0"
                  << std::endl;
        exit(1);
    }

    // resolve metric
    DistanceMetric resolved = resolve_metric(
        metric,
        {DistanceMetric::SumOfSquaredDistance,
         DistanceMetric::HistogramIntersection, DistanceMetric::CosineDistance},
        DistanceMetric::HistogramIntersection, "histogram");

    cv::Mat target;
    std::string target_path_str = std::string(target_path);
    target = cv::imread(target_path_str);
    // test if the read was successful
    if (target.data == NULL) {
        std::cout << "error: unable to read image " << target_path << std::endl;
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
            std::cout << "error: unable to read image " << image_path
                      << std::endl;
            exit(1);
        }

        // make image_histogram and compute/store distance
        cv::Mat image_histogram = compute_histogram(image, buckets);
        float distance;
        switch (resolved) {
        case DistanceMetric::SumOfSquaredDistance:
            distance = ssd_3d(target_histogram, image_histogram, buckets);
            break;
        case DistanceMetric::HistogramIntersection:
            distance = histogram_intersection_distance(
                target_histogram, image_histogram, buckets);
            break;
        case DistanceMetric::CosineDistance:
            distance =
                cosine_distance_3d(target_histogram, image_histogram, buckets);
            break;
        }
        results.emplace_back(distance, image_path);
    }
    std::sort(results.begin(), results.end());
    return results;
}

/*
    Compares a target image to a database of images in two ways. First the
   whole images, second a region of the center of the images. Uses a 3D
   RGB color histogram as the feature vector.

    @param target_path the target image path
    @param database_directory the directory path for the database of images
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @param edge_offset the scalar used to offset from edge for capturing
   central region of image
    @param uncropped_weight the weight given to the uncropped region
   (between 0.0f and 1.0f), the cropped_weight will be (1.0f -
   uncropped_weight)
    @param DistanceMetric the distance metric to use when comparing features
    @return the vector containing pairings between a float value and the
   string pathname
*/
std::vector<std::pair<float, std::string>>
multi_histogram(std::string_view target_path,
                std::string_view database_directory, int buckets,
                float edge_offset, float uncropped_weight,
                DistanceMetric metric) {
    // resolve metric
    DistanceMetric resolved = resolve_metric(
        metric,
        {DistanceMetric::SumOfSquaredDistance,
         DistanceMetric::HistogramIntersection, DistanceMetric::CosineDistance},
        DistanceMetric::HistogramIntersection, "multi_histogram");

    if (uncropped_weight < 0.0f || uncropped_weight > 1.0f) {
        std::cout << "error: cannot run multi_histogram with a "
                     "uncropped_weight of < 0.0f or > 1.0f"
                  << std::endl;
        exit(1);
    }
    if (edge_offset < 0.0f || edge_offset >= 0.5f) {
        std::cout << "error: cannot run multi_histogram with a "
                     "edge_offset of < 0.0f or >= 0.5f"
                  << std::endl;
        exit(1);
    }

    cv::Mat target = cv::imread(std::string(target_path));
    // test if the read was successful
    if (target.data == NULL) {
        std::cout << "error: unable to read image " << target_path << std::endl;
        exit(1);
    }

    // calculate region of interest (ROI)
    int target_x = target.cols * edge_offset;
    int target_y = target.rows * edge_offset;
    int target_width = target.cols - 2 * target_x;
    int target_height = target.rows - 2 * target_y;
    // validate ROI fits within the target
    if (target_x < 0 || target_y < 0 || target_x + target_width > target.cols ||
        target_y + target_height > target.rows) {
        std::cout
            << "error: region of interest does not fit within target image"
            << std::endl;
        exit(1);
    }
    cv::Rect target_roi(target_x, target_y, target_width, target_height);

    // compute both target histograms (uncropped and cropped)
    cv::Mat target_uncropped_histogram = compute_histogram(target, buckets);
    cv::Mat target_cropped = target(target_roi);
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
            std::cout << "error: unable to read image " << image_path
                      << std::endl;
            exit(1);
        }

        // calculate region of interest
        int image_x = image.cols * edge_offset;
        int image_y = image.rows * edge_offset;
        int image_width = image.cols - 2 * image_x;
        int image_height = image.rows - 2 * image_y;

        // validate ROI fits within the image
        if (image_x < 0 || image_y < 0 || image_x + image_width > image.cols ||
            image_y + image_height > image.rows) {
            std::cout << "error: region of interest does not fit within image"
                      << std::endl;
            exit(1);
        }
        cv::Rect image_roi(image_x, image_y, image_width, image_height);

        // compute both histograms for this image
        cv::Mat image_uncropped_histogram = compute_histogram(image, buckets);
        cv::Mat image_cropped = image(image_roi);
        cv::Mat image_cropped_histogram =
            compute_histogram(image_cropped, buckets);

        // compute both distances
        float uncropped_distance;
        float cropped_distance;
        switch (resolved) {
        case DistanceMetric::SumOfSquaredDistance:
            uncropped_distance = ssd_3d(target_uncropped_histogram,
                                        image_uncropped_histogram, buckets);
            cropped_distance = ssd_3d(target_cropped_histogram,
                                      image_cropped_histogram, buckets);
            break;
        case DistanceMetric::HistogramIntersection:
            uncropped_distance = histogram_intersection_distance(
                target_uncropped_histogram, image_uncropped_histogram, buckets);
            cropped_distance = histogram_intersection_distance(
                target_cropped_histogram, image_cropped_histogram, buckets);
            break;
        case DistanceMetric::CosineDistance:
            uncropped_distance = cosine_distance_3d(
                target_uncropped_histogram, image_uncropped_histogram, buckets);
            cropped_distance = cosine_distance_3d(
                target_cropped_histogram, image_cropped_histogram, buckets);
            break;
        }

        // weighted average
        float combined = uncropped_weight * uncropped_distance +
                         cropped_weight * cropped_distance;
        results.emplace_back(combined, image_path);
    }
    std::sort(results.begin(), results.end());
    return results;
}

/*
    Compares a target image to a database of images in two ways. First a
   color histogram, second a texture histogram as feature vectors. Uses
   histogram intersection as the distance metric for both histograms. Uses a
   Sobel gradient magnitude as texture feature. Histograms are weighted
   according to color_weight.

    @param target_path the target image path
    @param database_directory the directory path for the database of images
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @param color_weight the weight given to the color histogram (between
   0.0f and 1.0f), the texture_weight will be (1.0f - color_weight)
    @param DistanceMetric the distance metric to use when comparing features
    @return the vector containing pairings between a float value and the
   string pathname
*/
std::vector<std::pair<float, std::string>>
texture_color(std::string_view target_path, std::string_view database_directory,
              int buckets, float color_weight, DistanceMetric metric) {
    // resolve metric
    DistanceMetric resolved = resolve_metric(
        metric,
        {DistanceMetric::SumOfSquaredDistance,
         DistanceMetric::HistogramIntersection, DistanceMetric::CosineDistance},
        DistanceMetric::HistogramIntersection, "texture_color");

    if (color_weight < 0.0f || color_weight > 1.0f) {
        std::cout << "error: cannot run texture_color with a "
                     "color_weight of < 0.0f or > 1.0f"
                  << std::endl;
        exit(1);
    }

    // open target image
    cv::Mat target = cv::imread(std::string(target_path));
    // test if the read was successful
    if (target.data == NULL) {
        std::cout << "error: unable to read image " << target_path << std::endl;
        exit(1);
    }

    // generate 3D color histogram
    cv::Mat target_color_histogram = compute_histogram(target, buckets);

    // generate magnitude image
    cv::Mat target_sobel_x_3x3_output;
    cv::Mat target_sobel_y_3x3_output;
    sobelX3x3(target, target_sobel_x_3x3_output);
    sobelY3x3(target, target_sobel_y_3x3_output);

    cv::Mat target_magnitude;
    magnitude(target_sobel_x_3x3_output, target_sobel_y_3x3_output,
              target_magnitude);

    // generate magnitude histogram
    cv::Mat target_magnitude_histogram =
        compute_histogram(target_magnitude, buckets);

    float texture_weight = 1.0f - color_weight;
    std::vector<std::pair<float, std::string>> results;

    // loop through each image in data_base_directory
    for (const auto& entry :
         std::filesystem::directory_iterator(database_directory)) {

        std::string image_path = entry.path().string();
        cv::Mat image = cv::imread(image_path);
        if (image.data == NULL) {
            std::cout << "error: unable to read image " << image_path
                      << std::endl;
            exit(1);
        }

        // generate 3D color histogram
        cv::Mat image_color_histogram = compute_histogram(image, buckets);

        // generate magnitude image
        cv::Mat image_sobel_x_3x3_output;
        cv::Mat image_sobel_y_3x3_output;
        sobelX3x3(image, image_sobel_x_3x3_output);
        sobelY3x3(image, image_sobel_y_3x3_output);

        cv::Mat image_magnitude;
        magnitude(image_sobel_x_3x3_output, image_sobel_y_3x3_output,
                  image_magnitude);

        // generate magnitude histogram
        cv::Mat image_magnitude_histogram =
            compute_histogram(image_magnitude, buckets);

        // compute both distances
        float color_distance;
        float magnitude_distance;
        switch (resolved) {
        case DistanceMetric::SumOfSquaredDistance:
            color_distance =
                ssd_3d(target_color_histogram, image_color_histogram, buckets);
            magnitude_distance = ssd_3d(target_magnitude_histogram,
                                        image_magnitude_histogram, buckets);
            break;
        case DistanceMetric::HistogramIntersection:
            color_distance = histogram_intersection_distance(
                target_color_histogram, image_color_histogram, buckets);
            magnitude_distance = histogram_intersection_distance(
                target_magnitude_histogram, image_magnitude_histogram, buckets);
            break;
        case DistanceMetric::CosineDistance:
            color_distance = cosine_distance_3d(target_color_histogram,
                                                image_color_histogram, buckets);
            magnitude_distance = cosine_distance_3d(
                target_magnitude_histogram, image_magnitude_histogram, buckets);
            break;
        }

        // take average of the two histogram types
        float combined =
            color_weight * color_distance + texture_weight * magnitude_distance;
        results.emplace_back(combined, image_path);
    }
    std::sort(results.begin(), results.end());
    return results;
}

/*
    Compares a target to the other data in a csv using sum of squared
   differences.

    @param target the target image file name
    @param dnn_embeddings the csv file name
    @param DistanceMetric the distance metric to use when comparing features
    @return the vector containing pairings between a float value and the
   string pathname
*/
std::vector<std::pair<float, std::string>>
deep_network_embeddings(const char* target, const char* dnn_embeddings,
                        DistanceMetric metric) {
    // resolve metric
    DistanceMetric resolved = resolve_metric(
        metric,
        {DistanceMetric::SumOfSquaredDistance, DistanceMetric::CosineDistance},
        DistanceMetric::SumOfSquaredDistance, "deep_network_embeddings");

    // read image data from csv
    std::vector<char*> file_names;
    std::vector<std::vector<float>> data;
    int code = read_image_data_csv(dnn_embeddings, file_names, data, 0);
    if (code != 0) {
        std::cout << "error: read_image_data_csv failed" << std::endl;
        exit(1);
    }

    // get target index
    int target_index = 0;
    bool found = false;
    while (target_index < file_names.size()) {
        if (strcmp(file_names[target_index], target) == 0) {
            found = true;
            break;
        }
        target_index++;
    }
    if (!found) {
        std::cout << "error: " << target << " not found in " << dnn_embeddings
                  << std::endl;
        exit(1);
    }

    std::vector<std::pair<float, std::string>> results;

    // compare target data to all other image data in csv using resolved
    for (int i = 0; i < data.size(); i++) {
        double ssd = 0.0, dot = 0.0, target_norm = 0.0, image_norm = 0.0;
        for (int j = 0; j < data[i].size(); j++) {
            if (resolved == DistanceMetric::SumOfSquaredDistance) {
                float diff = data[target_index][j] - data[i][j];
                ssd += diff * diff;
            } else { // CosineDistance (only other supported)
                double tv = data[target_index][j], iv = data[i][j];
                dot += tv * iv;
                target_norm += tv * tv;
                image_norm += iv * iv;
            }
        }

        float distance;
        if (resolved == DistanceMetric::SumOfSquaredDistance) {
            distance = static_cast<float>(ssd);
        } else {
            if (target_norm == 0 || image_norm == 0) {
                distance = 1.0f;
            } else {
                distance =
                    static_cast<float>(1.0 - dot / (std::sqrt(target_norm) *
                                                    std::sqrt(image_norm)));
            }
        }
        results.emplace_back(distance, std::string(file_names[i]));
    }
    std::sort(results.begin(), results.end());
    return results;
}

/*
    Computes a 3D HSV histogram. H is [0, 180), S and V are [0, 256) due to
   OpenCV representation.

    @param image source image in BGR (will be converted to HSV internally)
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @return normalized 3D histogram (CV_32F, buckets × buckets × buckets)
*/
cv::Mat compute_hsv_histogram(cv::Mat& image, int buckets) {
    // convert to hsv
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // create histogram
    int sizes[] = {buckets, buckets, buckets};
    cv::Mat histogram(3, sizes, CV_32F, cv::Scalar(0));

    // loop through pixels quantizing and counting per value
    for (int i = 0; i < hsv.rows; i++) {
        cv::Vec3b* ptr = hsv.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hsv.cols; j++) {
            int h = ptr[j][0] * buckets / 180;
            int s = ptr[j][1] * buckets / 256;
            int v = ptr[j][2] * buckets / 256;

            // safety clamps (in case of rounding edge cases)
            if (h >= buckets)
                h = buckets - 1;
            if (s >= buckets)
                s = buckets - 1;
            if (v >= buckets)
                v = buckets - 1;

            histogram.at<float>(h, s, v) += 1.0f;
        }
    }
    histogram /= (hsv.rows * hsv.cols);
    return histogram;
}

/*
    Creates histograms between a target and a database and returns the
   comparisons.

    @param target_path the target image path
    @param database_directory the directory path for the database of images
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @param DistanceMetric the distance metric to use when comparing features
    @return the vector containing pairings between a float value and the
   string pathname
*/
std::vector<std::pair<float, std::string>>
hsv_histogram(std::string_view target_path, std::string_view database_directory,
              int buckets, DistanceMetric metric) {
    // resolve metric
    DistanceMetric resolved = resolve_metric(
        metric,
        {DistanceMetric::SumOfSquaredDistance,
         DistanceMetric::HistogramIntersection, DistanceMetric::CosineDistance},
        DistanceMetric::CosineDistance, "hsv_histogram");

    // read target
    cv::Mat target = cv::imread(std::string(target_path));
    if (target.data == NULL) {
        std::cout << "error: unable to read image " << target_path << std::endl;
        exit(1);
    }

    // compute hsv histogram
    cv::Mat target_hist = compute_hsv_histogram(target, buckets);

    std::vector<std::pair<float, std::string>> results;

    // loop through images in database and compare to target
    for (const auto& entry :
         std::filesystem::directory_iterator(database_directory)) {
        // read image
        std::string image_path = entry.path().string();
        cv::Mat image = cv::imread(image_path);
        if (image.data == NULL) {
            std::cout << "error: unable to read image " << image_path
                      << std::endl;
            exit(1);
        }

        // compute hsv histogram, calculate distance, then store distance
        cv::Mat image_hist = compute_hsv_histogram(image, buckets);

        // compute distance based upon resolved metric
        float distance;
        switch (resolved) {
        case DistanceMetric::SumOfSquaredDistance:
            distance = ssd_3d(target_hist, image_hist, buckets);
            break;
        case DistanceMetric::HistogramIntersection:
            distance = histogram_intersection_distance(target_hist, image_hist,
                                                       buckets);
            break;
        case DistanceMetric::CosineDistance:
            distance = cosine_distance_3d(target_hist, image_hist, buckets);
            break;
        }
        results.emplace_back(distance, image_path);
    }
    std::sort(results.begin(), results.end());
    return results;
}

/*
    Computes a 1D histogram of gradient orientations weighted by gradient
    magnitude. Orientations are in [0, π) since gradient sign is ignored for
    texture.

    @param sx_image Sobel X output (CV_16SC3)
    @param sy_image Sobel Y output (CV_16SC3)
    @param magnitude_image magnitude image (CV_8UC3)
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @return normalized 1D histogram (CV_32F, buckets x 1)
*/
cv::Mat compute_orientation_histogram(cv::Mat& sx_image, cv::Mat& sy_image,
                                      cv::Mat& magnitude_image, int buckets) {
    // create histogram cv::Mat
    cv::Mat histogram = cv::Mat::zeros(buckets, 1, CV_32F);
    double total_weight = 0.0;

    // loop through pixels in images
    for (int i = 0; i < sx_image.rows; i++) {
        cv::Vec3s* sx_ptr = sx_image.ptr<cv::Vec3s>(i);
        cv::Vec3s* sy_ptr = sy_image.ptr<cv::Vec3s>(i);
        cv::Vec3b* mag_ptr = magnitude_image.ptr<cv::Vec3b>(i);

        for (int j = 0; j < sx_image.cols; j++) {
            // average the three channels to get a single orientation per
            // pixel
            double sx_avg = (sx_ptr[j][0] + sx_ptr[j][1] + sx_ptr[j][2]) / 3.0;
            double sy_avg = (sy_ptr[j][0] + sy_ptr[j][1] + sy_ptr[j][2]) / 3.0;
            double mag_avg =
                (mag_ptr[j][0] + mag_ptr[j][1] + mag_ptr[j][2]) / 3.0;

            // atan2 returns (-π, π]; we map to [0, π) by taking absolute
            // value
            double angle = std::atan2(sy_avg, sx_avg);
            if (angle < 0)
                angle += CV_PI; // now in [0, π]
            if (angle >= CV_PI)
                angle -= 1e-9; // guard against == π

            int bin = static_cast<int>(angle * buckets / CV_PI);
            if (bin >= buckets) {
                bin = buckets - 1; // safety clamp
            }

            // add gradient magnitude instead of +=1 so a strong edge
            // contributes more than a near-zero gradient
            histogram.at<float>(bin) += static_cast<float>(mag_avg);
            total_weight += mag_avg;
        }
    }
    if (total_weight > 0) {
        histogram /= total_weight;
    }
    return histogram;
}

/*
    Helper, cosine distance for a 1D histogram

    @param a histogram a
    @param b histogram b
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @return cosine distance
 */
float cosine_distance_1d(cv::Mat& a, cv::Mat& b, int buckets) {
    double dot = 0.0;    // dot product
    double norm_a = 0.0; // sum of squares of a's components OR ||a||^2
    double norm_b = 0.0; // sum of squares for b's components
    for (int i = 0; i < buckets; i++) {
        float av = a.at<float>(i);
        float bv = b.at<float>(i);
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }
    // undefined / max distance. return completely dissimilar in this case
    if (norm_a == 0 || norm_b == 0) {
        return 1.0f;
    }
    double similarity = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    return static_cast<float>(1.0 - similarity); // 1 - similarity aka distance
}

/*
    Compare a target with images in database based upon orientation and
color. Computes orientation via SobelX, SobelY, and Gradient Magnitude.
Computes color via BGR.

    @param target_path the target image path
    @param database_directory the directory path for the database of images
    @param color_buckets the number of buckets the color vector should use
    @param orientation_buckets the number of buckets the orientation vector
should use
    @param color_weight the weight given to the color histogram (between
   0.0f and 1.0f), the texture_weight will be (1.0f - color_weight)
   @param DistanceMetric the distance metric to use when comparing features
    @return the vector containing pairings between a float value and the
string pathname
*/
std::vector<std::pair<float, std::string>>
orientation_color(std::string_view target_path,
                  std::string_view database_directory, int color_buckets,
                  int orientation_buckets, float color_weight,
                  DistanceMetric metric) {
    // resolve metric
    DistanceMetric resolved = resolve_metric(
        metric,
        {DistanceMetric::SumOfSquaredDistance,
         DistanceMetric::HistogramIntersection, DistanceMetric::CosineDistance},
        DistanceMetric::CosineDistance, "orientation_color");

    if (color_weight < 0.0f || color_weight > 1.0f) {
        std::cout << "error: cannot run orientation_color with color_weight < "
                     "0.0f or > 1.0f"
                  << std::endl;
        exit(1);
    }
    float orientation_weight = 1.0f - color_weight;

    // read target
    cv::Mat target = cv::imread(std::string(target_path));
    if (target.data == NULL) {
        std::cout << "error: unable to read image " << target_path << std::endl;
        exit(1);
    }

    // target color histogram
    cv::Mat target_color_histogram = compute_histogram(target, color_buckets);

    // target Sobel + magnitude + orientation histogram
    cv::Mat target_sx;
    cv::Mat target_sy;
    cv::Mat target_magnitude;
    sobelX3x3(target, target_sx);
    sobelY3x3(target, target_sy);
    magnitude(target_sx, target_sy, target_magnitude);
    cv::Mat target_orientation_histogram = compute_orientation_histogram(
        target_sx, target_sy, target_magnitude, orientation_buckets);

    std::vector<std::pair<float, std::string>> results;

    // loop through images comparing to target
    for (const auto& entry :
         std::filesystem::directory_iterator(database_directory)) {
        // read image
        std::string image_path = entry.path().string();
        cv::Mat image = cv::imread(image_path);
        if (image.data == NULL) {
            std::cout << "error: unable to read image " << image_path
                      << std::endl;
            exit(1);
        }

        // create image color histogram
        cv::Mat image_color_histogram = compute_histogram(image, color_buckets);

        // create image orientation histogram
        cv::Mat image_sx;
        cv::Mat image_sy;
        cv::Mat image_magnitude;
        sobelX3x3(image, image_sx);
        sobelY3x3(image, image_sy);
        magnitude(image_sx, image_sy, image_magnitude);
        cv::Mat image_orientation_histogram = compute_orientation_histogram(
            image_sx, image_sy, image_magnitude, orientation_buckets);

        // compute both distances
        float color_dist;
        float orient_dist;
        switch (resolved) {
        case DistanceMetric::SumOfSquaredDistance:
            color_dist = ssd_3d(target_color_histogram, image_color_histogram,
                                color_buckets);
            orient_dist =
                ssd_1d(target_orientation_histogram,
                       image_orientation_histogram, orientation_buckets);
            break;
        case DistanceMetric::HistogramIntersection:
            color_dist = histogram_intersection_distance(
                target_color_histogram, image_color_histogram, color_buckets);
            orient_dist = histogram_intersection_distance_1d(
                target_orientation_histogram, image_orientation_histogram,
                orientation_buckets);
            break;
        case DistanceMetric::CosineDistance:
            color_dist = cosine_distance_3d(
                target_color_histogram, image_color_histogram, color_buckets);
            orient_dist = cosine_distance_1d(target_orientation_histogram,
                                             image_orientation_histogram,
                                             orientation_buckets);
            break;
        }

        // combine via weights (default = 0.5f each)
        float combined =
            color_weight * color_dist + orientation_weight * orient_dist;
        results.emplace_back(combined, image_path);
    }
    std::sort(results.begin(), results.end());
    return results;
}

/*
    Compares a target image to a database of images using five feature
   vectors combined with equal 20% weighting:
      1. Whole-image BGR color histogram (histogram intersection distance)
      2. Whole-image gradient magnitude histogram (histogram intersection)
      3. Centered crop BGR color histogram (histogram intersection)
      4. Whole-image HSV color histogram (cosine distance)
      5. Whole-image gradient orientation histogram (cosine distance)

    @param target_path the target image path
    @param database_directory the directory path for the database of images
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @param edge_offset the scalar used to offset from edge for capturing
   central region of image (must be in [0, 0.5))
    @return the vector containing pairings between a float value and the
   string pathname
*/
std::vector<std::pair<float, std::string>>
combined_features(std::string_view target_path,
                  std::string_view database_directory, int color_buckets,
                  int orientation_buckets, float edge_offset) {
    if (edge_offset < 0.0f || edge_offset >= 0.5f) {
        std::cout << "error: cannot run combined_features with an "
                     "edge_offset of < 0.0f or >= 0.5f"
                  << std::endl;
        exit(1);
    }

    // read target
    cv::Mat target = cv::imread(std::string(target_path));
    if (target.data == NULL) {
        std::cout << "error: unable to read image " << target_path << std::endl;
        exit(1);
    }

    // --- target features ---

    // 1. whole-image BGR color histogram
    cv::Mat target_color_hist = compute_histogram(target, color_buckets);

    // 2. whole-image magnitude (texture) histogram
    cv::Mat target_sx, target_sy, target_magnitude;
    sobelX3x3(target, target_sx);
    sobelY3x3(target, target_sy);
    magnitude(target_sx, target_sy, target_magnitude);
    cv::Mat target_magnitude_hist =
        compute_histogram(target_magnitude, color_buckets);

    // 3. centered-crop BGR color histogram
    int target_x = target.cols * edge_offset;
    int target_y = target.rows * edge_offset;
    int target_width = target.cols - 2 * target_x;
    int target_height = target.rows - 2 * target_y;
    cv::Rect target_roi(target_x, target_y, target_width, target_height);
    cv::Mat target_cropped = target(target_roi);
    cv::Mat target_cropped_hist =
        compute_histogram(target_cropped, color_buckets);

    // 4. whole-image HSV color histogram
    cv::Mat target_hsv_hist = compute_hsv_histogram(target, color_buckets);

    // 5. whole-image gradient orientation histogram
    cv::Mat target_orientation_hist = compute_orientation_histogram(
        target_sx, target_sy, target_magnitude, orientation_buckets);

    std::vector<std::pair<float, std::string>> results;

    // each of the 5 features contributes 20% to final distance
    const float weight = 0.2f;

    // --- loop through database images ---
    for (const auto& entry :
         std::filesystem::directory_iterator(database_directory)) {
        std::string image_path = entry.path().string();
        cv::Mat image = cv::imread(image_path);
        if (image.data == NULL) {
            std::cout << "error: unable to read image " << image_path
                      << std::endl;
            exit(1);
        }

        // error on images too small for the ROI
        int image_x = image.cols * edge_offset;
        int image_y = image.rows * edge_offset;
        int image_width = image.cols - 2 * image_x;
        int image_height = image.rows - 2 * image_y;
        if (image_width <= 0 || image_height <= 0) {
            std::cout << "error: " << image_path << " (too small for ROI)"
                      << std::endl;
            exit(1);
        }
        cv::Rect image_roi(image_x, image_y, image_width, image_height);

        // 1. whole-image BGR color histogram
        cv::Mat image_color_hist = compute_histogram(image, color_buckets);

        // 2. magnitude histogram (also generates sx, sy for use in #5)
        cv::Mat image_sx, image_sy, image_magnitude;
        sobelX3x3(image, image_sx);
        sobelY3x3(image, image_sy);
        magnitude(image_sx, image_sy, image_magnitude);
        cv::Mat image_magnitude_hist =
            compute_histogram(image_magnitude, color_buckets);

        // 3. centered-crop color histogram
        cv::Mat image_cropped = image(image_roi);
        cv::Mat image_cropped_hist =
            compute_histogram(image_cropped, color_buckets);

        // 4. HSV color histogram
        cv::Mat image_hsv_hist = compute_hsv_histogram(image, color_buckets);

        // 5. gradient orientation histogram
        cv::Mat image_orientation_hist = compute_orientation_histogram(
            image_sx, image_sy, image_magnitude, orientation_buckets);

        // --- distances ---
        float color_dist = histogram_intersection_distance(
            target_color_hist, image_color_hist, color_buckets);
        float texture_dist = histogram_intersection_distance(
            target_magnitude_hist, image_magnitude_hist, color_buckets);
        float cropped_dist = histogram_intersection_distance(
            target_cropped_hist, image_cropped_hist, color_buckets);
        float hsv_dist =
            cosine_distance_3d(target_hsv_hist, image_hsv_hist, color_buckets);
        float orient_dist =
            cosine_distance_1d(target_orientation_hist, image_orientation_hist,
                               orientation_buckets);

        float combined = weight * (color_dist + texture_dist + cropped_dist +
                                   hsv_dist + orient_dist);
        results.emplace_back(combined, image_path);
    }
    std::sort(results.begin(), results.end());
    return results;
}
