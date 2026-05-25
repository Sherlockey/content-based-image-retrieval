/*
    Jonathon Davis
    2026-05-17
    Filters to be used with OpenCV cv::Mat data structures to modify the src.
*/

#include "filter.h"
#include "faceDetect.h"
#include <opencv2/opencv.hpp>

/*
    Applies a Lightness greyscale filter to a src and outputs it in dst

    @param src source image
    @param dst destination image
    @return error code, 0 on success, -1 on failure
*/
int greyscale(cv::Mat& src, cv::Mat& dst) {
    cv::Mat tmp;
    tmp.create(src.rows, src.cols, CV_8UC1);

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b* srcptr = src.ptr<cv::Vec3b>(i);
        uchar* tptr = tmp.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++) {
            short b = srcptr[j][0];
            short g = srcptr[j][1];
            short r = srcptr[j][2];

            // average method for greyscale
            // uchar y = (b + g + r) / 3;

            // lightness method for greyscale
            short max = std::max({b, g, r});
            short min = std::min({b, g, r});
            uchar y = (max + min) / 2;

            // max only method for greyscale
            // short max = std::max({b, g, r});
            // uchar y = max;

            // min only method for greyscale
            // short min = std::min({b, g, r});
            // uchar y = min;

            tptr[j] = y;
        }
    }
    tmp.copyTo(dst);
    return 0;
}

/*
    Implementation of a 5x5 blur filter using at<> method using integer
   approximation of Gaussian:

    1  2  4  2  1
    2  4  8  4  2
    4  8  16 8  4
    2  4  8  4  2
    1  2  4  2  1

    @param src source image
    @param dst destination image
    @return error code, 0 on success, -1 on failure
*/
int blur5x5_1(cv::Mat& src, cv::Mat& dst) {
    // valid convolution
    // leave outer row/col as the original image
    // allocates and copies the data to dst
    src.copyTo(dst);

    for (int i = 2; i < src.rows - 2; i++) {
        for (int j = 2; j < src.cols - 2; j++) {
            for (int k = 0; k < 3; k++) {
                dst.at<cv::Vec3b>(i, j)[k] =
                    (src.at<cv::Vec3b>(i - 2, j - 2)[k] * 1 +
                     src.at<cv::Vec3b>(i - 2, j - 1)[k] * 2 +
                     src.at<cv::Vec3b>(i - 2, j)[k] * 4 +
                     src.at<cv::Vec3b>(i - 2, j + 1)[k] * 2 +
                     src.at<cv::Vec3b>(i - 2, j + 2)[k] * 1 +

                     src.at<cv::Vec3b>(i - 1, j - 2)[k] * 2 +
                     src.at<cv::Vec3b>(i - 1, j - 1)[k] * 4 +
                     src.at<cv::Vec3b>(i - 1, j)[k] * 8 +
                     src.at<cv::Vec3b>(i - 1, j + 1)[k] * 4 +
                     src.at<cv::Vec3b>(i - 1, j + 2)[k] * 2 +

                     src.at<cv::Vec3b>(i, j - 2)[k] * 4 +
                     src.at<cv::Vec3b>(i, j - 1)[k] * 8 +
                     src.at<cv::Vec3b>(i, j)[k] * 16 +
                     src.at<cv::Vec3b>(i, j + 1)[k] * 8 +
                     src.at<cv::Vec3b>(i, j + 2)[k] * 4 +

                     src.at<cv::Vec3b>(i + 1, j - 2)[k] * 2 +
                     src.at<cv::Vec3b>(i + 1, j - 1)[k] * 4 +
                     src.at<cv::Vec3b>(i + 1, j)[k] * 8 +
                     src.at<cv::Vec3b>(i + 1, j + 1)[k] * 4 +
                     src.at<cv::Vec3b>(i + 1, j + 2)[k] * 2 +

                     src.at<cv::Vec3b>(i + 2, j - 2)[k] * 1 +
                     src.at<cv::Vec3b>(i + 2, j - 1)[k] * 2 +
                     src.at<cv::Vec3b>(i + 2, j)[k] * 4 +
                     src.at<cv::Vec3b>(i + 2, j + 1)[k] * 2 +
                     src.at<cv::Vec3b>(i + 2, j + 2)[k] * 1) /
                    100;
            }
        }
    }
    return 0;
}

/*
    Implementation of a 5x5 blur filter using separable 1x5 filters (vertical
   and horizontal) and the ptr method:

    1  2  4  2  1

    @param src source image
    @param dst destination image
    @return error code, 0 on success, -1 on failure
*/
int blur5x5_2(cv::Mat& src, cv::Mat& dst) {
    static cv::Mat tmp; // static for performance to avoid allocation overhead

    // allocate and initialize tmp
    src.copyTo(tmp);

    // run the horizontal 1x5 convolution [1 2 4 2 1]
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* tptr = tmp.ptr<cv::Vec3b>(i);
        for (int j = 2; j < src.cols - 2; j++) {
            for (int k = 0; k < 3; k++) {
                tptr[j][k] = ((sptr[j - 2][k] * 1) + (sptr[j - 1][k] * 2) +
                              (sptr[j][k] * 4) + (sptr[j + 1][k] * 2) +
                              (sptr[j + 2][k] * 1)) /
                             10;
            }
        }
    }

    // allocate and initialize dst
    tmp.copyTo(dst);

    // run the vertical 5x1 convolution [1 2 4 2 1]^t
    for (int i = 2; i < src.rows - 2; i++) {
        cv::Vec3b* tptrm2 = tmp.ptr<cv::Vec3b>(i - 2);
        cv::Vec3b* tptrm1 = tmp.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* tptr = tmp.ptr<cv::Vec3b>(i);
        cv::Vec3b* tptrp1 = tmp.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b* tptrp2 = tmp.ptr<cv::Vec3b>(i + 2);

        cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            for (int k = 0; k < 3; k++) {
                dptr[j][k] =
                    ((tptrm2[j][k] * 1) + (tptrm1[j][k] * 2) +
                     (tptr[j][k] * 4) + (tptrp1[j][k] * 2) + tptrp2[j][k] * 1) /
                    10;
            }
        }
    }
    return 0;
}

/*
    Applies a sepia filter to a src and outputs it in dst

    @param src source image
    @param dst destination image
    @return error code, 0 on success, -1 on failure
*/
int sepia(cv::Mat& src, cv::Mat& dst) {
    dst.create(src.rows, src.cols, src.type());

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b* srcptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* tmpptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int blue = srcptr[j][2] * 0.272f + srcptr[j][1] * 0.534f +
                       srcptr[j][0] * 0.131f;
            blue = std::clamp(blue, 0, 255);

            int green = srcptr[j][2] * 0.349f + srcptr[j][1] * 0.686f +
                        srcptr[j][0] * 0.168f;
            green = std::clamp(green, 0, 255);

            int red = srcptr[j][2] * 0.393f + srcptr[j][1] * 0.769f +
                      srcptr[j][0] * 0.189f;
            red = std::clamp(red, 0, 255);

            tmpptr[j] = cv::Vec3b(blue, green, red);
        }
    }

    return 0;
}

/*
    Applies a 3x3 sobel filter to a src and outputs it in dst by using separable
   1x3 filters where x is positive right:

   -1 0 1

   1
   2
   1

   -1 0 1
   -2 0 2
   -1 0 1

    @param src source image
    @param dst destination image
    @return error code, 0 on success, -1 on failure
*/
int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
    static cv::Mat tmp;
    tmp.create(src.rows, src.cols, CV_16SC3);
    dst.create(src.rows, src.cols, CV_16SC3);

    // run the horizontal 1x3 convolution [-1 0 1]
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3s* tptr = tmp.ptr<cv::Vec3s>(i);
        for (int j = 1; j < src.cols - 1; j++) {
            for (int k = 0; k < 3; k++) {
                tptr[j][k] = (sptr[j - 1][k] * -1) + (sptr[j][k] * 0) +
                             (sptr[j + 1][k] * 1);
            }
        }
    }

    // run the vertical 3x1 convolution [1 2 1]^t
    for (int i = 1; i < src.rows - 1; i++) {
        cv::Vec3s* tptrm1 = tmp.ptr<cv::Vec3s>(i - 1);
        cv::Vec3s* tptr = tmp.ptr<cv::Vec3s>(i);
        cv::Vec3s* tptrp1 = tmp.ptr<cv::Vec3s>(i + 1);
        cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);
        for (int j = 0; j < src.cols; j++) {
            for (int k = 0; k < 3; k++) {
                dptr[j][k] = ((tptrm1[j][k] * 1) + (tptr[j][k] * 2) +
                              (tptrp1[j][k] * 1)) /
                             4;
            }
        }
    }
    return 0;
}

/*
    Applies a 3x3 sobel filter to a src and outputs it in dst by using separable
   1x3 filters where y is positive up:

   1 2 1

    1
    0
   -1

    1  2  1
    0  0  0
   -1 -2 -1

    @param src source image
    @param dst destination image
    @return error code, 0 on success, -1 on failure
*/
int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
    static cv::Mat tmp;
    tmp.create(src.rows, src.cols, CV_16SC3);
    dst.create(src.rows, src.cols, CV_16SC3);

    // run the horizontal 1x3 convolution [1 2 1]
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3s* tptr = tmp.ptr<cv::Vec3s>(i);
        for (int j = 1; j < src.cols - 1; j++) {
            for (int k = 0; k < 3; k++) {
                tptr[j][k] = (sptr[j - 1][k] * 1) + (sptr[j][k] * 2) +
                             (sptr[j + 1][k] * 1);
            }
        }
    }

    // run the vertical 3x1 convolution [1 0 -1]^t
    for (int i = 1; i < src.rows - 1; i++) {
        cv::Vec3s* tptrm1 = tmp.ptr<cv::Vec3s>(i - 1);
        cv::Vec3s* tptr = tmp.ptr<cv::Vec3s>(i);
        cv::Vec3s* tptrp1 = tmp.ptr<cv::Vec3s>(i + 1);
        cv::Vec3s* dptr = dst.ptr<cv::Vec3s>(i);
        for (int j = 0; j < src.cols; j++) {
            for (int k = 0; k < 3; k++) {
                dptr[j][k] = ((tptrm1[j][k] * 1) + (tptr[j][k] * 0) +
                              (tptrp1[j][k] * -1)) /
                             4;
            }
        }
    }
    return 0;
}

/*
    Generates a gradient magnitude image from x and y Sobel images where:
    I = sqrt(sx*sx + sy*sy)

    @param sx sobelx source image
    @param sy sobely source image
    @param dst destination image
    @return error code, 0 on success, -1 on failure
*/
int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
    if (sx.rows != sy.rows || sx.cols != sy.cols) {
        return -1;
    }

    dst.create(sx.rows, sx.cols, CV_8UC3);
    cv::Mat tmp;
    tmp.create(sx.rows, sx.cols, CV_16SC3);

    for (int i = 0; i < sx.rows; i++) {
        cv::Vec3s* sxptr = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s* syptr = sy.ptr<cv::Vec3s>(i);
        cv::Vec3s* tptr = tmp.ptr<cv::Vec3s>(i);
        for (int j = 0; j < sx.cols; j++) {
            for (int k = 0; k < 3; k++) {
                // I = sqrt(sx*sx + sy*sy)
                int mag = std::sqrt((int)sxptr[j][k] * sxptr[j][k] +
                                    (int)syptr[j][k] * syptr[j][k]);
                tptr[j][k] = mag;
            }
        }
    }
    cv::convertScaleAbs(tmp, dst);
    return 0;
}

/*
    Blurs, then quantizes an image according to @param levels

    @param src source image
    @param dst destination image
    @param levels the levels of quantization, higher = more colors, lower = less
    @return error code, 0 on success, -1 on failure
*/
int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels) {
    if (levels == 0) {
        return -1;
    }
    dst.create(src.rows, src.cols, src.type());

    blur5x5_2(src, dst);

    for (int i = 0; i < dst.rows; i++) {
        cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < dst.cols; j++) {
            for (int k = 0; k < 3; k++) {
                // apply quantization based upon levels (buckets)
                int x = dptr[j][k];
                float b = 255.0f / (levels - 1);
                int xt = x / b;
                int xf = xt * b;
                dptr[j][k] = xf;
            }
        }
    }
    return 0;
}

/*
    Game Boy style quantization to four shades of green

    #071821 hex, 15, 56, 15 BGR -- darkest green,
    #306850 hex, 48, 98, 48 BGR -- dark green,
    #86c06c hex, 139, 172, 15 BGR -- light green,
    #e0f8cf hex, 155, 188, 15 BGR -- lightest green,

    @param src source image
    @param dst destination image
    @return error code, 0 on success, -1 on failure
*/
int gameBoy(cv::Mat& src, cv::Mat& dst) {
    cv::Mat tmp;
    cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    dst.create(src.rows, src.cols, src.type());

    for (int i = 0; i < tmp.rows; i++) {
        uchar* tptr = tmp.ptr<uchar>(i);
        cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < tmp.cols; j++) {
            int x = tptr[j];
            // using x, apply four shades of green according to evenly
            // distributed buckets
            if (x < 64) {
                dptr[j] = cv::Vec3b(15, 56, 15);
            } else if (x < 128) {
                dptr[j] = cv::Vec3b(48, 98, 48);
            } else if (x < 192) {
                dptr[j] = cv::Vec3b(139, 172, 15);
            } else {
                dptr[j] = cv::Vec3b(155, 188, 15);
            }
        }
    }
    return 0;
}

/*
    3x3 Emboss filter
    -1 -1  0
    -1  0  1
     0  1  1

    @param src source image
    @param dst destination image
    @return error code, 0 on success, -1 on failure
*/
int emboss(cv::Mat& src, cv::Mat& dst) {
    cv::Mat tmp;
    tmp.create(src.rows, src.cols, src.type());

    for (int i = 1; i < src.rows - 1; i++) {
        cv::Vec3b* rm1 = src.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* r0 = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* rp1 = src.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b* tptr = tmp.ptr<cv::Vec3b>(i);
        for (int j = 1; j < src.cols - 1; j++) {
            for (int k = 0; k < 3; k++) {
                int sum = rm1[j - 1][k] * -1 + rm1[j][k] * -1 +
                          rm1[j + 1][k] * 0 + r0[j - 1][k] * -1 + r0[j][k] * 0 +
                          r0[j + 1][k] * 1 + rp1[j - 1][k] * 0 + rp1[j][k] * 1 +
                          rp1[j + 1][k] * 1;
                // boost by 128 then clamp
                sum += 128;
                tptr[j][k] = (uchar)std::clamp(sum, 0, 255);
            }
        }
    }
    tmp.copyTo(dst);
    return 0;
}

/*
    Pixellation filter

    @param src source image
    @param dst destination image
    @param size to divide width and height by when downsampling
    @return error code, 0 on success, -1 on failure
 */
int pixelate(cv::Mat& src, cv::Mat& dst, int size) {
    if (size < 1) {
        return -1;
    }

    // downsample with linear then upsample with nearest-neighbor
    cv::Mat small_img;
    int small_w = std::max(1, src.cols / size);
    int small_h = std::max(1, src.rows / size);
    cv::resize(src, small_img, cv::Size(small_w, small_h), 0, 0,
               cv::INTER_LINEAR);
    cv::resize(small_img, dst, src.size(), 0, 0, cv::INTER_NEAREST);
    return 0;
}

/*
    Filter using facial recognition to censor someone's face

    @param src source image
    @param dst destination image
    @param size to divide width and height by when downsampling (in pixelate)
    @return error code, 0 on success, -1 on failure
*/
int censorFace(cv::Mat& src, cv::Mat& dst, int size) {
    std::vector<cv::Rect> faces;
    cv::Mat grey;
    src.copyTo(dst);
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY, 0);
    detectFaces(grey, faces);

    // only pixelate what is inside faces
    for (const auto& face : faces) {
        // clamp the rect to frame bounds in case the face is at the edge
        cv::Rect rect = face & cv::Rect(0, 0, dst.cols, dst.rows);
        if (rect.area() == 0) {
            continue;
        }

        cv::Mat face_region = dst(rect);
        cv::Mat censored;
        pixelate(face_region, censored, size);
        censored.copyTo(dst(rect));
    }
    return 0;
}

/*
    Applies a CRT filter to the image via: scanlines, BGR subpixel pattern, and
   vignette

    @param src source image
    @param dst destination image
    @param scanline_strength: 0 = no scanlines, 1 = full black
    @param subpixel_strength: 0 = no RGB stripes, 1 = full mask
    @param vignette_strength: 0 = no vignette, 1 = corners fully black
    @param brightness_boost: boost to offset overall dimming ~2.0f = original
    @return error code, 0 on success, -1 on failure
*/
int crt(cv::Mat& src, cv::Mat& dst, float scanline_strength,
        float subpixel_strength, float vignette_strength,
        float brightness_boost) {
    dst.create(src.rows, src.cols, src.type());

    // vignette geometry
    float vx = src.cols * 0.5f;
    float vy = src.rows * 0.5f;
    float max_dist2 = vx * vx + vy * vy;

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);

        float scan_factor = (i % 2 == 0) ? 1.0f : (1.0f - scanline_strength);

        for (int j = 0; j < src.cols; j++) {
            // subpixel mask where each col emphasizes one of B, G, R in order
            float mask[3] = {1.0f, 1.0f, 1.0f};
            int col_mod3 = j % 3;
            for (int k = 0; k < 3; k++) {
                if (k != col_mod3) {
                    mask[k] = 1.0f - subpixel_strength;
                }
            }

            // vignette factor
            float dx = j - vx;
            float dy = i - vy;
            float v =
                1.0f - vignette_strength * ((dx * dx + dy * dy) / max_dist2);

            // apply all
            float combined = scan_factor * v * brightness_boost;
            for (int k = 0; k < 3; k++) {
                float val = sptr[j][k] * combined * mask[k];
                dptr[j][k] = (uchar)std::clamp((int)val, 0, 255);
            }
        }
    }
    return 0;
}

/*
    Applies a Negative filter to the image

    @param src source image
    @param dst destination image
    @return error code, 0 on success, -1 on failure
*/
int negative(cv::Mat& src, cv::Mat& dst) {
    dst.create(src.rows, src.cols, src.type());

    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            for (int k = 0; k < 3; k++) {
                // apply negative by subtracting max value
                dptr[j][k] = 255 - sptr[j][k];
            }
        }
    }
    return 0;
}

/*
    3x3 Corner filter
     1 -2  1
    -2  4 -2
     1 -2  1

    @param src source image
    @param dst destination image
    @return error code, 0 on success, -1 on failure
*/
int corner(cv::Mat& src, cv::Mat& dst) {
    cv::Mat tmp;
    tmp.create(src.rows, src.cols, src.type());

    for (int i = 1; i < src.rows - 1; i++) {
        cv::Vec3b* rm1 = src.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* r0 = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* rp1 = src.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b* tptr = tmp.ptr<cv::Vec3b>(i);
        for (int j = 1; j < src.cols - 1; j++) {
            for (int k = 0; k < 3; k++) {
                int sum = rm1[j - 1][k] * 1 + rm1[j][k] * -2 +
                          rm1[j + 1][k] * 1 + r0[j - 1][k] * -2 + r0[j][k] * 4 +
                          r0[j + 1][k] * -2 + rp1[j - 1][k] * 1 +
                          rp1[j][k] * -2 + rp1[j + 1][k] * 1;
                // boost by 128 then clamp
                sum += 128;
                tptr[j][k] = (uchar)std::clamp(sum, 0, 255);
            }
        }
    }
    tmp.copyTo(dst);
    return 0;
}

/*
    3x3 Bilinear filter
    1 2 1
    2 4 2
    1 2 1

    @param src source image
    @param dst destination image
    @return error code, 0 on success, -1 on failure
*/
int bilinear(cv::Mat& src, cv::Mat& dst) {
    cv::Mat tmp;
    tmp.create(src.rows, src.cols, src.type());

    for (int i = 1; i < src.rows - 1; i++) {
        cv::Vec3b* rm1 = src.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b* r0 = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* rp1 = src.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b* tptr = tmp.ptr<cv::Vec3b>(i);
        for (int j = 1; j < src.cols - 1; j++) {
            for (int k = 0; k < 3; k++) {
                int sum = rm1[j - 1][k] * 1 + rm1[j][k] * 2 +
                          rm1[j + 1][k] * 1 + r0[j - 1][k] * 2 + r0[j][k] * 4 +
                          r0[j + 1][k] * 2 + rp1[j - 1][k] * 1 + rp1[j][k] * 2 +
                          rp1[j + 1][k] * 1;
                // divide by amount in system and clamp
                sum /= 16;
                tptr[j][k] = (uchar)std::clamp(sum, 0, 255);
            }
        }
    }
    tmp.copyTo(dst);
    return 0;
}
