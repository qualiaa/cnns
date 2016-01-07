#pragma once

#include <armadillo>
#include <tuple>

namespace loader {

using byte = unsigned char;

using Label = byte;
using Image = arma::Mat<byte>;

std::tuple<std::vector<Image>, // train images
           std::vector<Label>, // train classes
           std::vector<Image>, // test images
           std::vector<Label>> // test classes
load_MNIST_data();
}
