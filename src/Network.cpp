#include "Network.hpp"

#include <random>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <iostream>

namespace cnn {

using value_type = Network::value_type;
using Vec = Network::Vec;
using Mat = Network::Mat;
using MatList = Network::MatList;
using VecList = Network::VecList;

const std::function<Vec(Vec)> Network::activation = Network::sigmoid;
const std::function<Vec(Vec)> Network::dActivation = Network::dSigmoid;
const std::function<value_type(Vec const&, Vec const&)> Network::cost
    = Network::mse;
const std::function<Vec(Vec const&, Vec const&)> Network::dCost
    = Network::dMSE;

template <typename T = double>
T randf(T a = 0.0, T b = 1.0)
{
    static std::ranlux48 source(std::random_device{}());
    return std::uniform_real_distribution<T>(a,b)(source);
}


Network::Network(std::vector<int> const& sizes)
    : layers{sizes.size()}
{
    for (unsigned l = 0; l+1 < layers; ++l) {
        const int n_in = sizes[l], n_out = sizes[l+1];

        weights.emplace_back(n_out, n_in, arma::fill::randu);
        biases.emplace_back(n_out, arma::fill::randu);
    }
}

std::tuple<VecList,VecList> Network::feed_forward(Vec const& x) const
{
    VecList activations{x};
    VecList zs;

    for (unsigned l = 0; l+1 < layers; ++l) {
        auto const& W = weights[l];
        auto const& b = biases[l];
        auto const& x = activations.back();
        auto const z = W * x + b;
        zs.push_back(z);
        activations.push_back(activation(z));
    }
    return std::tuple<VecList,VecList>(activations, zs);
}

Vec Network::classify(Vec const& x) const
{
    return std::get<0>(feed_forward(x)).back();
}

VecList Network::back_propagate(Vec const& x)
{
    return {x};
}

void Network::train(Mat const& data, Mat const& classes, double learning_rate)
{
    const unsigned n_instances = data.n_rows;
    std::stringstream stream;
    if (data.n_rows != classes.n_rows) {
        stream << "All training instances must have associated output data:\n"
               << "Provided " << data.n_rows << " instances, "
                              << classes.n_rows << " desired outputs\n";
        throw std::invalid_argument(stream.str());
    }
    if (data.n_cols != weights.front().n_cols) {
        stream << "Instance data must fill input layer:\n"
               << "Provided " << data.n_cols << " data, have "
                              << weights.front().n_cols << "input nodes\n";
        throw std::invalid_argument(stream.str());
    }
    if (classes.n_cols != weights.back().n_rows) {
        stream << "Desired output data must match output layer:\n"
               << "Provided " << classes.n_cols << " desired outputs, have "
                              << weights.back().n_rows << " output nodes\n";
        throw std::invalid_argument(stream.str());
    }

    learning_rate /= n_instances;

    MatList delta_weights;
    VecList delta_biases;
    for (unsigned l = 0; l+1 < layers; ++l) {
        delta_weights.emplace_back(size(weights[l]), arma::fill::zeros);
        delta_biases.emplace_back(size(biases[l]), arma::fill::zeros);
    }

    for (unsigned i = 0; i < n_instances; ++i) {
        const Vec x = data.row(i).t();
        const Vec y = classes.row(i).t();

        VecList activations, zs;
        std::tie(activations, zs) = feed_forward(x);

        const Vec delta_cost = dCost(y, activations.back());
        
        Vec running_derivative = delta_cost % dActivation(zs.back()); // delta

        // back-propagation + gradient descent
        for (int l = layers - 1; l > 0; --l) {
            // update deltas with dC wrt weights and biases in l
            delta_weights[l-1] += arma::dot(running_derivative, activations[l]);
            delta_biases[l-1] += running_derivative;

            // update delta
            if (l > 1) {
                running_derivative = (weights[l-1].t() * running_derivative) % dActivation(zs[l-2]);
            }

            // modify weights according to gradient descent
            weights[l-1] -= learning_rate * delta_weights[l-1];
            biases[l-1]  -= learning_rate * delta_biases[l-1];

        }
    }
}
}
