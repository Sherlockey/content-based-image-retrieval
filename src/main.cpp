#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string_view>

int main(int argc, char* argv[]) {
    std::vector<std::string_view> args(argv + 1, argv + argc);

    // get command line args
    if (args.size() != 5) {
        std::cout << "Usage: " << argv[0] << " <target image filename>"
                  << " <database filepath>"
                  << " <'baseline'/'histogram'/'multi-histogram'/"
                     "'texture-color'/'deep-network-embeddings'>"
                  << " <'sum-of-squared-distance'/'histogram-intersection'/"
                     "'rgb-histogram-intersection'/'custom'>"
                  << " <number of output images>" << std::endl;
        exit(1);
    }

    // parse command line args and set values
    // target image T
    std::string_view target_path = args[0];
    if (!std::filesystem::exists(target_path)) {
        std::cout << target_path << " is not a valid file" << std::endl;
        exit(1);
    }
    if (!target_path.ends_with(".png") && !target_path.ends_with(".jpg")) {
        std::cout << target_path << " is not a .png or .jpg file" << std::endl;
        exit(1);
    }
    // database B
    std::string_view database_directory = args[1];
    if (!std::filesystem::is_directory(database_directory)) {
        std::cout << database_directory << " is not a valid directory"
                  << std::endl;
        exit(1);
    }
    // method of computing features F
    std::string_view feature_method = args[2];
    if (feature_method != "baseline" && feature_method != "histogram" &&
        feature_method != "multi-histogram" &&
        feature_method != "texture-color" &&
        feature_method != "deep-network-embeddings") {
        std::cout
            << feature_method
            << " must be one of: <'baseline'/'histogram'/'multi-histogram'/"
               "'texture-color'/'deep-network-embeddings'>"
            << std::endl;
        exit(1);
    }
    // distance metric for comparisons D
    std::string_view distance_metric = args[3];
    if (distance_metric != "sum-of-squared-distance" &&
        distance_metric != "histogram-intersection" &&
        distance_metric != "rgb-histogram-intersection" &&
        distance_metric != "custom") {
        std::cout << distance_metric
                  << " must be one of: "
                     "<'sum-of-squared-distance'/'histogram-intersection'/"
                     "'rgb-histogram-intersection'/'custom'>"
                  << std::endl;
        exit(1);
    }
    // # of desired output images N
    int num_out_images = std::stoi(std::string(args[4]));

    // compute the features Ft on the target image T
    cv::Mat target;
    std::string target_path_str = std::string(target_path);
    target = cv::imread(target_path_str);

    // // test if the read was successful
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

    // calculate sum-of-squared-difference
    int region_dim = 7;
    int half_dim = region_dim / 2;
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
        for (int i = 0; i < region_dim; i++) {
            cv::Vec3b* target_ptr = target.ptr<cv::Vec3b>(target_start_row + i);
            cv::Vec3b* image_ptr = image.ptr<cv::Vec3b>(image_start_row + i);
            for (int j = 0; j < region_dim; j++) {
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

    int num_to_print =
        std::min(num_out_images, static_cast<int>(results.size()));
    std::cout << "Top " << num_to_print << " matches for " << target_path << ":"
              << std::endl;
    for (int i = 0; i < num_to_print; i++) {
        std::cout << i + 1 << ". " << results[i].second
                  << " (SSD: " << results[i].first << ")" << std::endl;
    }

    // compute the features {Fi} on all of the images in B

    // compute the dinstance of T from all of the images in B using the
    // distance

    // metric D(Ft, Fi)

    // sort the images in B according to their distance from T and return
    // the best N matches

    return 0;
}
