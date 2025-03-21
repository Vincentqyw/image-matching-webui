
#include <b64/encode.h>
#include <fstream>
#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

// base64 to image
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>

/// Parameters used in the API
struct APIParams {
    /// A list of images, base64 encoded
    std::vector<std::string> data;

    /// The maximum number of keypoints to detect for each image
    std::vector<int> max_keypoints;

    /// The timestamps of the images
    std::vector<std::string> timestamps;

    /// Whether to convert the images to grayscale
    bool grayscale;

    /// The height and width of each image
    std::vector<std::vector<int>> image_hw;

    /// The type of feature detector to use
    int feature_type;

    /// The rotations of the images
    std::vector<double> rotates;

    /// The scales of the images
    std::vector<double> scales;

    /// The reference points of the images
    std::vector<std::vector<float>> reference_points;

    /// Whether to binarize the descriptors
    bool binarize;
};

/**
 * @brief Contains the results of a keypoint detector.
 *
 * @details Stores the keypoints and descriptors for each image.
 */
class KeyPointResults {
      public:
    KeyPointResults() {
    }

    /**
     * @brief Constructor.
     *
     * @param kp The keypoints for each image.
     */
    KeyPointResults(const std::vector<std::vector<cv::KeyPoint>>& kp,
                    const std::vector<cv::Mat>& desc)
        : keypoints(kp), descriptors(desc) {
    }

    /**
     * @brief Append keypoints to the result.
     *
     * @param kpts The keypoints to append.
     */
    inline void append_keypoints(std::vector<cv::KeyPoint>& kpts) {
        keypoints.emplace_back(kpts);
    }

    /**
     * @brief Append descriptors to the result.
     *
     * @param desc The descriptors to append.
     */
    inline void append_descriptors(cv::Mat& desc) {
        descriptors.emplace_back(desc);
    }

    /**
     * @brief Get the keypoints.
     *
     * @return The keypoints.
     */
    inline std::vector<std::vector<cv::KeyPoint>> get_keypoints() {
        return keypoints;
    }

    /**
     * @brief Get the descriptors.
     *
     * @return The descriptors.
     */
    inline std::vector<cv::Mat> get_descriptors() {
        return descriptors;
    }

      private:
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;
    std::vector<std::vector<float>> scores;
};

/**
 * @brief Decodes a base64 encoded string.
 *
 * @param base64 The base64 encoded string to decode.
 * @return The decoded string.
 */
std::string base64_decode(const std::string& base64) {
    using namespace boost::archive::iterators;
    using It = transform_width<binary_from_base64<std::string::const_iterator>, 8, 6>;

    // Find the position of the last non-whitespace character
    auto end = base64.find_last_not_of(" \t\n\r");
    if (end != std::string::npos) {
        // Move one past the last non-whitespace character
        end += 1;
    }

    // Decode the base64 string and return the result
    return std::string(It(base64.begin()), It(base64.begin() + end));
}

/**
 * @brief Decodes a base64 string into an OpenCV image
 *
 * @param base64 The base64 encoded string
 * @return The decoded OpenCV image
 */
cv::Mat base64_to_image(const std::string& base64) {
    // Decode the base64 string
    std::string decodedStr = base64_decode(base64);

    // Decode the image
    std::vector<uchar> data(decodedStr.begin(), decodedStr.end());
    cv::Mat img = cv::imdecode(data, cv::IMREAD_GRAYSCALE);

    // Check for errors
    if (img.empty()) {
        throw std::runtime_error("Failed to decode image");
    }

    return img;
}

/**
 * @brief Encodes an OpenCV image into a base64 string
 *
 * This function takes an OpenCV image and encodes it into a base64 string.
 * The image is first encoded as a PNG image, and then the resulting
 * bytes are encoded as a base64 string.
 *
 * @param img The OpenCV image
 * @return The base64 encoded string
 *
 * @throws std::runtime_error if the image is empty or encoding fails
 */
std::string image_to_base64(cv::Mat& img) {
    if (img.empty()) {
        throw std::runtime_error("Failed to read image");
    }

    // Encode the image as a PNG
    std::vector<uchar> buf;
    if (!cv::imencode(".png", img, buf)) {
        throw std::runtime_error("Failed to encode image");
    }

    // Encode the bytes as a base64 string
    using namespace boost::archive::iterators;
    using It =
        base64_from_binary<transform_width<std::vector<uchar>::const_iterator, 6, 8>>;
    std::string base64(It(buf.begin()), It(buf.end()));

    // Pad the string with '=' characters to a multiple of 4 bytes
    base64.append((3 - buf.size() % 3) % 3, '=');

    return base64;
}

/**
 * @brief Callback function for libcurl to write data to a string
 *
 * This function is used as a callback for libcurl to write data to a string.
 * It takes the contents, size, and nmemb as parameters, and writes the data to
 * the string.
 *
 * @param contents The data to write
 * @param size The size of the data
 * @param nmemb The number of members in the data
 * @param s The string to write the data to
 * @return The number of bytes written
 */
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t newLength = size * nmemb;
    try {
        // Resize the string to fit the new data
        s->resize(s->size() + newLength);
    } catch (std::bad_alloc& e) {
        // If there's an error allocating memory, return 0
        return 0;
    }

    // Copy the data to the string
    std::copy(static_cast<const char*>(contents),
              static_cast<const char*>(contents) + newLength,
              s->begin() + s->size() - newLength);
    return newLength;
}

// Helper functions

/**
 * @brief Helper function to convert a type to a Json::Value
 *
 * This function takes a value of type T and converts it to a Json::Value.
 * It is used to simplify the process of converting a type to a Json::Value.
 *
 * @param val The value to convert
 * @return The converted Json::Value
 */
template <typename T> Json::Value toJson(const T& val) {
    return Json::Value(val);
}

/**
 * @brief Converts a vector to a Json::Value
 *
 * This function takes a vector of type T and converts it to a Json::Value.
 * Each element in the vector is appended to the Json::Value array.
 *
 * @param vec The vector to convert to Json::Value
 * @return The Json::Value representing the vector
 */
template <typename T> Json::Value vectorToJson(const std::vector<T>& vec) {
    Json::Value json(Json::arrayValue);
    for (const auto& item : vec) {
        json.append(item);
    }
    return json;
}

/**
 * @brief Converts a nested vector to a Json::Value
 *
 * This function takes a nested vector of type T and converts it to a
 * Json::Value. Each sub-vector is converted to a Json::Value array and appended
 * to the main Json::Value array.
 *
 * @param vec The nested vector to convert to Json::Value
 * @return The Json::Value representing the nested vector
 */
template <typename T>
Json::Value nestedVectorToJson(const std::vector<std::vector<T>>& vec) {
    Json::Value json(Json::arrayValue);
    for (const auto& subVec : vec) {
        json.append(vectorToJson(subVec));
    }
    return json;
}

/**
 * @brief Converts the APIParams struct to a Json::Value
 *
 * This function takes an APIParams struct and converts it to a Json::Value.
 * The Json::Value is a JSON object with the following fields:
 * - data: a JSON array of base64 encoded images
 * - max_keypoints: a JSON array of integers, max number of keypoints for each
 * image
 * - timestamps: a JSON array of timestamps, one for each image
 * - grayscale: a JSON boolean, whether to convert images to grayscale
 * - image_hw: a nested JSON array, each sub-array contains the height and width
 * of an image
 * - feature_type: a JSON integer, the type of feature detector to use
 * - rotates: a JSON array of doubles, the rotation of each image
 * - scales: a JSON array of doubles, the scale of each image
 * - reference_points: a nested JSON array, each sub-array contains the
 * reference points of an image
 * - binarize: a JSON boolean, whether to binarize the descriptors
 *
 * @param params The APIParams struct to convert
 * @return The Json::Value representing the APIParams struct
 */
Json::Value paramsToJson(const APIParams& params) {
    Json::Value json;
    json["data"] = vectorToJson(params.data);
    json["max_keypoints"] = vectorToJson(params.max_keypoints);
    json["timestamps"] = vectorToJson(params.timestamps);
    json["grayscale"] = toJson(params.grayscale);
    json["image_hw"] = nestedVectorToJson(params.image_hw);
    json["feature_type"] = toJson(params.feature_type);
    json["rotates"] = vectorToJson(params.rotates);
    json["scales"] = vectorToJson(params.scales);
    json["reference_points"] = nestedVectorToJson(params.reference_points);
    json["binarize"] = toJson(params.binarize);
    return json;
}

template <typename T> cv::Mat jsonToMat(Json::Value json) {
    int rows = json.size();
    int cols = json[0].size();

    // Create a single array to hold all the data.
    std::vector<T> data;
    data.reserve(rows * cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data.push_back(static_cast<T>(json[i][j].asInt()));
        }
    }

    // Create a cv::Mat object that points to the data.
    cv::Mat mat(rows, cols, CV_8UC1,
                data.data());  // Change the type if necessary.
    // cv::Mat mat(cols, rows,CV_8UC1, data.data());  // Change the type if
    // necessary.

    return mat;
}

/**
 * @brief Decodes the response of the server and prints the keypoints
 *
 * This function takes the response of the server, a JSON string, and decodes
 * it. It then prints the keypoints and draws them on the original image.
 *
 * @param response The response of the server
 * @return The keypoints and descriptors
 */
KeyPointResults decode_response(const std::string& response, bool viz = true) {
    Json::CharReaderBuilder builder;
    Json::CharReader* reader = builder.newCharReader();

    Json::Value jsonData;
    std::string errors;

    // Parse the JSON response
    bool parsingSuccessful = reader->parse(
        response.c_str(), response.c_str() + response.size(), &jsonData, &errors);
    delete reader;

    if (!parsingSuccessful) {
        // Handle error
        std::cout << "Failed to parse the JSON, errors:" << std::endl;
        std::cout << errors << std::endl;
        return KeyPointResults();
    }

    KeyPointResults kpts_results;

    // Iterate over the images
    for (const auto& jsonItem : jsonData) {
        auto jkeypoints = jsonItem["keypoints"];
        auto jkeypoints_orig = jsonItem["keypoints_orig"];
        auto jdescriptors = jsonItem["descriptors"];
        auto jscores = jsonItem["scores"];
        auto jimageSize = jsonItem["image_size"];
        auto joriginalSize = jsonItem["original_size"];
        auto jsize = jsonItem["size"];

        std::vector<cv::KeyPoint> vkeypoints;
        std::vector<float> vscores;

        // Iterate over the keypoints
        int counter = 0;
        for (const auto& keypoint : jkeypoints_orig) {
            if (counter < 10) {
                // Print the first 10 keypoints
                std::cout << keypoint[0].asFloat() << ", " << keypoint[1].asFloat()
                          << std::endl;
            }
            counter++;
            // Convert the Json::Value to a cv::KeyPoint
            vkeypoints.emplace_back(
                cv::KeyPoint(keypoint[0].asFloat(), keypoint[1].asFloat(), 0.0));
        }

        if (viz && jsonItem.isMember("image_orig")) {
            auto jimg_orig = jsonItem["image_orig"];
            cv::Mat img = jsonToMat<uchar>(jimg_orig);
            cv::imwrite("viz_image_orig.jpg", img);

            // Draw keypoints on the image
            cv::Mat imgWithKeypoints;
            cv::drawKeypoints(img, vkeypoints, imgWithKeypoints, cv::Scalar(0, 0, 255));

            // Write the image with keypoints
            std::string filename = "viz_image_orig_keypoints.jpg";
            cv::imwrite(filename, imgWithKeypoints);
        }

        // Iterate over the descriptors
        cv::Mat descriptors = jsonToMat<uchar>(jdescriptors);
        kpts_results.append_keypoints(vkeypoints);
        kpts_results.append_descriptors(descriptors);
    }
    return kpts_results;
}
