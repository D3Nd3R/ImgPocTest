#include <QCoreApplication>

#include <thread>
#include <vector>

#include "perf_test.hpp"

std::vector<cv::Size> makeResulutions(){
    cv::Size base(320,180);
    std::vector<cv::Size> ret;
    ret.reserve(15);

    for(size_t ii = 1; ii <= 15; ++ii)
    {
        ret.emplace_back(base.width*ii, base.height*ii);
    }
    return ret;
}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    auto resulutions = makeResulutions();

    for (const auto &res : resulutions)
        for (size_t ii = 1; ii <= std::thread::hardware_concurrency(); ++ii){
            matInternalForEach(res,ii);
            parallelForTest(res,ii);

        }

    return a.exec();
}
