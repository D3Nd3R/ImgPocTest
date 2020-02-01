#include <QCoreApplication>

#include <thread>

#include "perf_test.hpp"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    for (size_t ii = 1; ii <= std::thread::hardware_concurrency(); ++ii){
        matInternalForEach(cv::Size(1920,1080),ii);
    }


    return a.exec();
}
