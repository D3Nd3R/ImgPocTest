#ifndef PERF_TEST_HPP
#define PERF_TEST_HPP
#include <opencv2/core/core.hpp>


void matInternalForEach(cv::Size imgSize, size_t threadNum);

void parallelForTest(cv::Size imgSize, size_t threadNum);

#endif // PERF_TEST_HPP
