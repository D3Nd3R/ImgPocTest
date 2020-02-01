#include "perf_test.hpp"

#include <random>
#include <array>
#include <chrono>
#include <iostream>
#include <thread>

const size_t pivotPointSize = 5;

using timePoint = decltype (std::chrono::high_resolution_clock::now());

class Profiler{
public:
    void tic(){
        start = std::chrono::high_resolution_clock::now();
    }
    /**
     * @brief toc
     * @return number of milliseconds from start
     */
    long long toc(){
        stop = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    }
private:
    timePoint start;
    timePoint stop;
};

std::string curThreadId() noexcept
{
    std::stringstream ss;
    ss << std::this_thread::get_id();
    return ss.str();
}



struct PivotPoint {
    cv::Point2d pt;
    double val;

    PivotPoint(cv::Point2d _pt, double _val): pt(_pt), val(_val){}
    PivotPoint() = default;
};

std::array<PivotPoint,pivotPointSize> makePivotPoints(cv::Size sz){

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> widthDis(1., sz.width);
    std::uniform_real_distribution<> heightDis(1., sz.height);
    std::uniform_real_distribution<> valDis(1., 1000.);
    std::array<PivotPoint,pivotPointSize> points;

    for (size_t ii = 0; ii < pivotPointSize; ++ii){
        points[ii] = {cv::Point2d(widthDis(gen),widthDis(gen)),valDis(gen)};
    }

    return points;
}


void matInternalForEach(cv::Size imgSize, size_t threadNum)
{
    auto points = makePivotPoints(imgSize);

    auto surfCalcSeq = [&points](double &val, const int *position)-> void{

        //std::cout << "surf.forEach thread_ID= " << curThreadId().c_str() << std::endl;

        std::array<double, pivotPointSize> dists;

        int row = position[0];
        int col = position[1];

        cv::Point2d curPos(col,row);
        double total_dist {0.f};

        for (size_t ii = 0; ii < dists.size(); ++ii){
            double curDist = cv::norm(points[ii].pt - curPos);
            dists[ii] = curDist;
            total_dist += curDist;
        }

        for (size_t i = 0; i < points.size(); ++i) {
            val += (dists[i] / total_dist) * points[i].val;
        }
    };

    cv::Mat_<double> surf = cv::Mat::zeros(imgSize, CV_64F);

    cv::setNumThreads(threadNum);
    //int thNumReal = cv::getThreadNum();

    Profiler prf;

    prf.tic();

    surf.forEach(surfCalcSeq);

    std::cout <<"threadsNum = " << cv::getNumThreads() << "   " << prf.toc() << " ms" << std::endl;
}
