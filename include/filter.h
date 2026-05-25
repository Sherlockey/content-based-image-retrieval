/*
    Jonathon Davis
    2026-05-19
    CS 5330 Computer Vision

    Include file for filter.cpp.
    Filters to be used with OpenCV cv::Mat data structures to modify the src.
*/

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

// prototypes
int greyscale(cv::Mat& src, cv::Mat& dst);
int blur5x5_1(cv::Mat& src, cv::Mat& dst);
int blur5x5_2(cv::Mat& src, cv::Mat& dst);
int sepia(cv::Mat& src, cv::Mat& dst);
int sobelX3x3(cv::Mat& src, cv::Mat& dst);
int sobelY3x3(cv::Mat& src, cv::Mat& dst);
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels);
int gameBoy(cv::Mat& src, cv::Mat& dst);
int emboss(cv::Mat& src, cv::Mat& dst);
int pixelate(cv::Mat& src, cv::Mat& dst, int size);
int censorFace(cv::Mat& src, cv::Mat& dst, int size);
int crt(cv::Mat& src, cv::Mat& dst, float scanline_strength = 0.35f,
        float subpixel_strength = 0.95, float vignette_strength = 0.65f,
        float brightness_boost = 1.50f);
int negative(cv::Mat& src, cv::Mat& dst);
int corner(cv::Mat& src, cv::Mat& dst);
int bilinear(cv::Mat& src, cv::Mat& dst);

#endif // FILTER_H
