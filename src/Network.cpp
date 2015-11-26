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
    , weights(layers)
    , biases(layers)
{
    for (unsigned l = 1; l < layers; ++l) {
        const int n_in = sizes[l-1], n_out = sizes[l];

        weights[l] = Mat(n_out, n_in, arma::fill::randu);
        std::cout << l << ": " << n_out << std::endl;
        biases[l] = Vec(n_out, arma::fill::randu);
    }
}

std::tuple<VecList,VecList> Network::feed_forward(Vec const& x) const
{
    VecList activations(layers, x);
    VecList weighted_inputs(layers);

    for (unsigned l = 1; l < layers; ++l) {
        auto const& W = weights[l];
        auto const& b = biases[l];
        auto const& x = activations[l-1];
        auto const z = W * x + b;

        weighted_inputs[l] = z;
        activations[l] = activation(z);
    }
    return std::tuple<VecList,VecList>(activations, weighted_inputs);
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
    if (data.n_cols != weights[1].n_cols) {
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

    MatList delta_weights(layers);
    VecList delta_biases(layers);

    for (unsigned l = 1; l < layers; ++l) {
        delta_weights[l] = Mat(size(weights[l]), arma::fill::zeros);
        delta_biases[l] = Vec(size(biases[l]), arma::fill::zeros);
    }

    for (unsigned i = 0; i < n_instances; ++i) {
        const Vec x = data.row(i).t();
        const Vec y = classes.row(i).t();

        VecList activations, zs;
        std::tie(activations, zs) = feed_forward(x);

        const Vec dCda = dCost(y, activations.back());
        
        Vec running_derivative = dCda % dActivation(zs.back()); // delta

        // back-propagation + gradient descent
        for (int l = layers - 1; l > 1; --l) {
            // update deltas with dC wrt weights and biases in l
            delta_weights[l] += running_derivative * activations[l-1].t();
            std::cout << delta_weights[l];
            delta_biases[l] += running_derivative;

            // update delta
            running_derivative = (weights[l].t() * running_derivative) % dActivation(zs[l-1]);

            // modify weights according to gradient descent
            weights[l] -= learning_rate * delta_weights[l];
            biases[l]  -= learning_rate * delta_biases[l];

        }
        std::cout << "Wanted: " << y;
        std::cout << "Found: " << activations.back();
    }
}
}
