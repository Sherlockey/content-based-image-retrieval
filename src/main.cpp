/*
    Jonathon Davis
    2026-05-24
    Given a database of images and a target image, find images in the data with
   similar content
*/

#include "matching.hpp"
#include <filesystem>

/*
    Helper to print results

    @param results the vector containing a pairing between the float value and
   the string pathname
    @param target_path target image file path
    @param num_out_images the number of images to print
*/
void print_results(std::vector<std::pair<float, std::string>> results,
                   std::string_view target_path, int num_out_images) {
    int num_to_print =
        std::min(num_out_images, static_cast<int>(results.size()));
    std::cout << "Top " << num_to_print << " matches for " << target_path << ":"
              << std::endl;
    for (int i = 1; i < num_to_print + 1 && i < results.size(); i++) {
        std::cout << i << ". " << results[i].second
                  << " (Value: " << results[i].first << ")" << std::endl;
    }
}

/*
    Program entry point

    @param argc argument count with how many command line arguments provided
    @param argv argument vector is an array of strings with command line args
    @return exit status / return code sent back to OS
*/
int main(int argc, char* argv[]) {
    std::vector<std::string_view> args(argv + 1, argv + argc);

    // get command line args
    if (args.size() != 5) {
        std::cout << "Usage: " << argv[0] << " <target image filename>"
                  << " <database filepath>"
                  << " <'baseline'/'histogram'/'multi-histogram'/"
                     "'texture-color'/'deep-network-embeddings'/"
                     "'hsv-histogram'/'orientation-color'/'combined-features'>"
                  << " <'sum-of-squared-distance'/'histogram-intersection'/"
                     "'cosine-distance'>"
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
        feature_method != "deep-network-embeddings" &&
        feature_method != "hsv-histogram" &&
        feature_method != "orientation-color" &&
        feature_method != "combined-features") {
        std::cout
            << feature_method
            << " must be one of: <'baseline'/'histogram'/'multi-histogram'/"
               "'texture-color'/'deep-network-embeddings'/'hsv-histogram'/"
               "'orientation-color'/'combined-features'>"
            << std::endl;
        exit(1);
    }
    // distance metric for comparisons D
    std::string_view distance_metric = args[3];
    if (distance_metric != "sum-of-squared-distance" &&
        distance_metric != "histogram-intersection" &&
        distance_metric != "cosine-distance") {
        std::cout << distance_metric
                  << " must be one of: "
                     "<'sum-of-squared-distance'/'histogram-intersection'/"
                     "'cosine-distance'>"
                  << std::endl;
        exit(1);
    }
    DistanceMetric metric = parse_distance_metric(distance_metric);

    // # of desired output images N
    int num_out_images = std::stoi(std::string(args[4]));

    // execute based upon feature_method
    if (feature_method == "baseline") {
        int region_dimension = 7;
        auto results =
            baseline(target_path, database_directory, region_dimension);

        std::cout << "---Baseline---" << std::endl;
        print_results(results, target_path, num_out_images);
    } else if (feature_method == "histogram") {
        int buckets = 16;

        auto results =
            histogram(target_path, database_directory, buckets, metric);

        std::cout << "---Histogram---" << std::endl;
        print_results(results, target_path, num_out_images);
    } else if (feature_method == "multi-histogram") {
        int buckets = 16;
        float edge_offset = 0.25f;
        float uncropped_weight = 0.5f;

        auto results = multi_histogram(target_path, database_directory, buckets,
                                       edge_offset, uncropped_weight, metric);

        std::cout << "---Mutli-Histogram---" << std::endl;
        print_results(results, target_path, num_out_images);
    } else if (feature_method == "texture-color") {
        int buckets = 16;
        float color_weight = 0.5f;
        auto results = texture_color(target_path, database_directory, buckets,
                                     color_weight, metric);
        std::cout << "---Texture and Color---" << std::endl;
        print_results(results, target_path, num_out_images);
    } else if (feature_method == "deep-network-embeddings") {
        std::filesystem::path p(target_path);
        std::string file_name = p.filename().string();
        const char* c = file_name.c_str();
        const char* dnn_embeddings_path = "./data/ResNet18_olym.csv";

        auto results = deep_network_embeddings(c, dnn_embeddings_path);

        std::cout << "---Deep Network Embeddings---" << std::endl;
        print_results(results, target_path, num_out_images);
    } else if (feature_method == "hsv-histogram") {
        int buckets = 16;

        auto results =
            hsv_histogram(target_path, database_directory, buckets, metric);

        std::cout << "---HSV Histogram---" << std::endl;
        print_results(results, target_path, num_out_images);
    } else if (feature_method == "orientation-color") {
        int color_buckets = 16;
        int orientation_buckets = 9;
        float color_weight = 0.5f;

        auto results =
            orientation_color(target_path, database_directory, color_buckets,
                              orientation_buckets, color_weight, metric);

        std::cout << "---Orientation Color---" << std::endl;
        print_results(results, target_path, num_out_images);
    } else if (feature_method == "combined-features") {
        int color_buckets = 16;
        int orientation_buckets = 9;
        float edge_offset = 0.25f;

        auto results =
            combined_features(target_path, database_directory, color_buckets,
                              orientation_buckets, edge_offset);

        std::cout << "---Combined Features---" << std::endl;
        print_results(results, target_path, num_out_images);
    }
}
