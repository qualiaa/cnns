#pragma once

#include <cmath>
#include <functional>
#include <vector>
#include <armadillo>

namespace cnn {

class Network {
public:
    using value_type = double;
    using Vec = arma::Col<value_type>;
    using Mat = arma::Mat<value_type>;
    using VecList = std::vector<Vec>;
    using MatList = std::vector<Mat>;

    static const std::function<Vec(Vec)> activation;
    static const std::function<Vec(Vec)> dActivation;
    static const std::function<value_type(Vec const&, Vec const&)> cost;
    static const std::function<Vec(Vec const&, Vec const&)> dCost;

private:
    const size_t layers;
    MatList weights;
    VecList biases;

public:
    Network(std::vector<int> const& size);

    Vec classify(Vec const& x) const;

    void train(Mat const& data, Mat const& classes, double learning_rate = 1.0);

private:
    std::tuple<VecList,VecList> feed_forward(Vec const& x) const;
    VecList back_propagate(Vec const& x);

    static Vec sigmoid(Vec z) {
        return z.transform([] (auto z) {
                return 1 / (1 + std::exp(-z));
            });
    }

    static Vec dSigmoid(Vec z) {
        return z.transform([] (auto z) {
                auto ez = std::exp(-z);
                auto s = 1/(1 + ez);
                return ez * s * s;
            });
    }

    static value_type mse(Vec const& y, Vec const& a) {
        using namespace arma;
        return sum(square(y - a));
        
    }

    static Vec dMSE(Vec const& y, Vec const& a) {
        return a - y;
    }
};
}
