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

VecList Network::classify(VecList const& X) const
{
    VecList outputs;
    std::transform(X.begin(), X.end(), std::back_inserter(outputs),
                   [this] (auto const& x) { return classify(x); });
    return outputs;
    
}

std::tuple<MatList, VecList> Network::backpropagate(VecList const& activations,
                                                    VecList const& zs,
                                                    Vec const& y)
{
    MatList delta_weights(layers);
    VecList delta_biases(layers);

    const Vec dCda = dCost(y, activations.back());
    Vec running_derivative = dCda % dActivation(zs.back()); // delta

    // back-propagation + gradient descent
    for (int l = layers - 1; l > 1; --l) {
        // update deltas with dC wrt weights and biases in l
        delta_weights[l] = running_derivative * activations[l-1].t();
        delta_biases[l] = running_derivative;

        // update delta
        running_derivative = (weights[l].t() * running_derivative) % dActivation(zs[l-1]);
    }

    return std::make_tuple(delta_weights, delta_biases);
}

void Network::train(Mat const& data, Mat const& classes, double learning_rate)
{
    const unsigned n_instances = data.n_cols;
    std::stringstream stream;
    if (data.n_cols != classes.n_cols) {
        stream << "All training instances must have associated output data:\n"
               << "Provided " << data.n_cols << " instances, "
                              << classes.n_cols << " labels\n";
        throw std::invalid_argument(stream.str());
    }
    if (data.n_rows != weights[1].n_cols) {
        stream << "Instance data must fill input layer:\n"
               << "Provided " << data.n_rows << " data, have "
                              << weights.front().n_cols << "input nodes\n";
        throw std::invalid_argument(stream.str());
    }
    if (classes.n_rows != weights.back().n_rows) {
        stream << "Desired output data must match output layer:\n"
               << "Provided " << classes.n_rows << " desired outputs, have "
                              << weights.back().n_rows << " output nodes\n";
        throw std::invalid_argument(stream.str());
    }

    learning_rate /= n_instances;

    for (unsigned i = 0; i < n_instances; ++i) {
        const Vec x = data.col(i);
        const Vec y = classes.col(i);

        VecList activations, zs;
        std::tie(activations, zs) = feed_forward(x);

        MatList delta_weights;
        VecList delta_biases;
        std::tie(delta_weights, delta_biases) = backpropagate(activations, zs, y);

        for (int l = layers - 1; l > 1; --l) {
            weights[l] -= learning_rate * delta_weights[l];
            biases[l]  -= learning_rate * delta_biases[l];
        }
    }
}
}
