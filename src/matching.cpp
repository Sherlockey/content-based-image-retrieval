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
    @param edge_offset the scalar used to offset from edge for capturing central
   region of image
    @param uncropped_weight the weight given to the uncropped region (between
   0.0f and 1.0f), the cropped_weight will be (1.0f - uncropped_weight)
    @return the vector containing pairings between a float value and the string
    pathname
*/
std::vector<std::pair<float, std::string>>
multi_histogram(std::string_view target_path,
                std::string_view database_directory, int buckets,
                float edge_offset, float uncropped_weight) {

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
        std::cout << "error: unable to read image" << target_path << std::endl;
        exit(1);
    }

    // calculate region of interest (ROI)
    int target_x = target.cols * edge_offset;
    int target_y = target.rows * edge_offset;
    int target_width = target.cols - 2 * target_x;
    int target_height = target.rows - 2 * target_y;
    // validate ROI fits within the target
    if (target_x < 0 || target_y < 0 || target_x + target_width > target.cols ||
        target_x + target_height > target.rows) {
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
            std::cout << "error: unable to read image" << image_path
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
            image_x + image_height > image.rows) {
            std::cout
                << "error: region of interest does not fit within image image"
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

/*
    Compares a target image to a database of images in two ways. First a color
   histogram, second a texture histogram as feature vectors. Uses histogram
   intersection as the distance metric for both histograms. Uses a Sobel
   gradient magnitude as texture feature. Histograms are weighted according to
   color_weight.

    @param target_path the target image path
    @param database_directory the directory path for the database of images
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @param color_weight the weight given to the color histogram (between
   0.0f and 1.0f), the texture_weight will be (1.0f - color_weight)
    @return the vector containing pairings between a float value and the string
    pathname
*/
std::vector<std::pair<float, std::string>>
texture_color(std::string_view target_path, std::string_view database_directory,
              int buckets, float color_weight) {
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
        std::cout << "error: unable to read image" << target_path << std::endl;
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
            std::cout << "error: unable to read image" << image_path
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
        float color_distance = histogram_intersection_distance(
            target_color_histogram, image_color_histogram, buckets);
        float magnitude_distance = histogram_intersection_distance(
            target_magnitude_histogram, image_magnitude_histogram, buckets);

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
    @return the vector containing pairings between a float value and the string
    pathname
*/
std::vector<std::pair<float, std::string>>
deep_network_embeddings(const char* target, const char* dnn_embeddings) {
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

    // compare target data to all other image data in csv using SSD
    for (int i = 0; i < data.size(); i++) {
        float ssd = 0.0; // sum of squared difference
        for (int j = 0; j < data[i].size(); j++) {
            // do stuff
            float difference = data[target_index][j] - data[i][j];
            ssd += difference * difference;
        }
        results.emplace_back(ssd, std::string(file_names[i]));
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
        // undefined / max distance. return completely dissimilar in this case
        return 1.0f;
    }
    double similarity = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    return static_cast<float>(1.0 - similarity); // 1 - similarity, aka distance
}

/*
    Creates histograms between a target and a database and returns the
   comparisons.

    @param target_path the target image path
    @param database_directory the directory path for the database of images
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @return the vector containing pairings between a float value and the string
    pathname
*/
std::vector<std::pair<float, std::string>>
hsv_histogram(std::string_view target_path, std::string_view database_directory,
              int buckets) {
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
        float distance = cosine_distance_3d(target_hist, image_hist, buckets);
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
            // average the three channels to get a single orientation per pixel
            double sx_avg = (sx_ptr[j][0] + sx_ptr[j][1] + sx_ptr[j][2]) / 3.0;
            double sy_avg = (sy_ptr[j][0] + sy_ptr[j][1] + sy_ptr[j][2]) / 3.0;
            double mag_avg =
                (mag_ptr[j][0] + mag_ptr[j][1] + mag_ptr[j][2]) / 3.0;

            // atan2 returns (-π, π]; we map to [0, π) by taking absolute value
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
float cosine_distance_1d(const cv::Mat& a, const cv::Mat& b, int buckets) {
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
    Compare a target with images in database based upon orientation and color.
   Computes orientation via SobelX, SobelY, and Gradient Magnitude. Computes
   color via BGR.

    @param target_path the target image path
    @param database_directory the directory path for the database of images
    @param buckets the number of buckets the feature vector should use
   (quantize)
    @param color_weight the weight given to the color histogram (between
   0.0f and 1.0f), the texture_weight will be (1.0f - color_weight)
    @return the vector containing pairings between a float value and the string
pathname
*/
std::vector<std::pair<float, std::string>>
orientation_color(std::string_view target_path,
                  std::string_view database_directory, int buckets,
                  float color_weight) {
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
    cv::Mat target_color_histogram = compute_histogram(target, buckets);

    // target Sobel + magnitude + orientation histogram
    cv::Mat target_sx;
    cv::Mat target_sy;
    cv::Mat target_magnitude;
    sobelX3x3(target, target_sx);
    sobelY3x3(target, target_sy);
    magnitude(target_sx, target_sy, target_magnitude);
    cv::Mat target_orientation_histogram = compute_orientation_histogram(
        target_sx, target_sy, target_magnitude, buckets);

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
        cv::Mat image_color_histogram = compute_histogram(image, buckets);

        // create image orientation histogram
        cv::Mat image_sx;
        cv::Mat image_sy;
        cv::Mat image_magnitude;
        sobelX3x3(image, image_sx);
        sobelY3x3(image, image_sy);
        magnitude(image_sx, image_sy, image_magnitude);
        cv::Mat image_orientation_histogram = compute_orientation_histogram(
            image_sx, image_sy, image_magnitude, buckets);

        // cosine distance for each, 3D for color, 1D for distance
        float color_dist = cosine_distance_3d(target_color_histogram,
                                              image_color_histogram, buckets);
        float orient_dist = cosine_distance_1d(
            target_orientation_histogram, image_orientation_histogram, buckets);

        // combine via weights (default = 0.5f each)
        float combined =
            color_weight * color_dist + orientation_weight * orient_dist;
        results.emplace_back(combined, image_path);
    }
    std::sort(results.begin(), results.end());
    return results;
}
