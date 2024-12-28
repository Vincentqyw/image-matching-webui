#include <curl/curl.h>
#include <opencv2/opencv.hpp>
#include "helper.h"

int main() {
    std::string img_path =
        "../../../datasets/sacre_coeur/mapping_rot/02928139_3448003521_rot45.jpg";
    cv::Mat original_img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

    if (original_img.empty()) {
        throw std::runtime_error("Failed to decode image");
    }

    // Convert the image to Base64
    std::string base64_img = image_to_base64(original_img);

    // Convert the Base64 back to an image
    cv::Mat decoded_img = base64_to_image(base64_img);
    cv::imwrite("decoded_image.jpg", decoded_img);
    cv::imwrite("original_img.jpg", original_img);

    // The images should be identical
    if (cv::countNonZero(original_img != decoded_img) != 0) {
        std::cerr << "The images are not identical" << std::endl;
        return -1;
    } else {
        std::cout << "The images are identical!" << std::endl;
    }

    // construct params
    APIParams params{.data = {base64_img},
                     .max_keypoints = {100, 100},
                     .timestamps = {"0", "1"},
                     .grayscale = {0},
                     .image_hw = {{480, 640}, {240, 320}},
                     .feature_type = 0,
                     .rotates = {0.0f, 0.0f},
                     .scales = {1.0f, 1.0f},
                     .reference_points = {{1.23e+2f, 1.2e+1f},
                                          {5.0e-1f, 3.0e-1f},
                                          {2.3e+2f, 2.2e+1f},
                                          {6.0e-1f, 4.0e-1f}},
                     .binarize = {1}};

    KeyPointResults kpts_results;

    // Convert the parameters to JSON
    Json::Value jsonData = paramsToJson(params);
    std::string url = "http://127.0.0.1:8001/v1/extract";
    Json::StreamWriterBuilder writer;
    std::string output = Json::writeString(writer, jsonData);

    CURL* curl;
    CURLcode res;
    std::string readBuffer;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if (curl) {
        struct curl_slist* hs = NULL;
        hs = curl_slist_append(hs, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hs);
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, output.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        res = curl_easy_perform(curl);

        if (res != CURLE_OK)
            fprintf(
                stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        else {
            // std::cout << "Response from server: " << readBuffer << std::endl;
            kpts_results = decode_response(readBuffer);
        }
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();

    return 0;
}
