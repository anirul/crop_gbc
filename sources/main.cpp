#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <cassert>
#include <iostream>
#include <opencv2/opencv.hpp>

ABSL_FLAG(std::string, input_file, "", "Input file image to be processed.");
ABSL_FLAG(
    std::string, output_file, "output.png", "Output file for the GBC file.");

std::uint8_t QuantizeGreyPixel(
    std::uint16_t value, const std::vector<std::uint8_t>& pixels)
{
    std::uint8_t out_val = 0;
    std::uint8_t max_distance = std::numeric_limits<std::uint8_t>::max();
    for (const auto& pixel : pixels)
    {
        std::uint16_t distance = std::abs(pixel - value);
        if (distance < max_distance)
        {
            out_val = pixel;
            max_distance = distance;
        }
    }
    return out_val;
}

cv::Mat QuantizeGreyMat(
    const cv::Mat& mat, const std::vector<std::uint8_t>& pixels)
{
    cv::Mat out_img = mat;
    for (int y = 0; y < mat.rows; ++y)
    {
        for (int x = 0; x < mat.cols; ++x)
        {
            out_img.at<std::uint8_t>(y, x) =
                QuantizeGreyPixel(mat.at<std::uint8_t>(y, x), pixels);
        }
    }
    return out_img;
}

void DitheringGreyErrorPixel(float error, cv::Mat& mat, int x, int y)
try
{
    if (x < mat.cols - 1)
    {
        float val = mat.at<std::int16_t>(y, x + 1);
        mat.at<std::int16_t>(y, x + 1) = val + error * 7.0 / 16.0;
    }
    if (x > 0 && y < mat.rows - 1)
    {
        float val = mat.at<std::int16_t>(y + 1, x - 1);
        mat.at<std::int16_t>(y + 1, x - 1) = val + error * 3.0 / 16.0;
    }
    if (y < mat.rows - 1)
    {
        float val = mat.at<std::int16_t>(y + 1, x);
        mat.at<std::int16_t>(y + 1, x) = val + error * 5.0 / 16.0;
    }
    if (x < mat.cols - 1 && y < mat.rows - 1)
    {
        float val = mat.at<std::int16_t>(y + 1, x + 1);
        mat.at<std::int16_t>(y + 1, x + 1) = val + error * 1.0 / 16.0;
    }
}
catch (...)
{
    std::cerr << "Error " << x << ", " << y << " : could not be done?\n";
}

cv::Mat DitheringGreyMat(
    const cv::Mat& mat, const std::vector<std::uint8_t>& pixels)
{
    cv::Mat error_img;
    cv::Mat out_img = mat;
    mat.convertTo(error_img, CV_16SC1);
    for (int y = 0; y < mat.rows; ++y)
    {
        for (int x = 0; x < mat.cols; ++x)
        {
            std::int16_t mat_pixel = error_img.at<std::int16_t>(y, x);
            std::uint8_t quantize_pixel = QuantizeGreyPixel(mat_pixel, pixels);
            float error = static_cast<float>(mat_pixel) -
                          static_cast<float>(quantize_pixel);
            out_img.at<std::uint8_t>(y, x) = quantize_pixel;
            DitheringGreyErrorPixel(error, error_img, x, y);
        }
    }
    return out_img;
}

int main(int ac, char** av)
try
{
    absl::ParseCommandLine(ac, av);
    std::string input_path = absl::GetFlag(FLAGS_input_file);
    std::string output_path = absl::GetFlag(FLAGS_output_file);
    cv::Size gbc_size = cv::Size(128, 112);
    if (input_path.empty())
    {
        throw std::runtime_error("no input file.");
    }
    std::cout << "Input file             : [" << input_path << "]\n";
    cv::Mat img = cv::imread(input_path.c_str());
    cv::Mat resized;
    cv::resize(img, resized, gbc_size);
    cv::cvtColor(resized, img, cv::COLOR_BGR2GRAY);
    cv::Mat final = DitheringGreyMat(img, {0, 85, 170, 255});
    std::cout << "Writing output to file : [" << output_path << "]\n";
    cv::imwrite(output_path.c_str(), final);
    return 0;
}
catch (std::exception ex)
{
    std::cerr << "Exception: " << ex.what() << std::endl;
}
