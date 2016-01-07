#include <algorithm>
#include <random>

#include "Network.hpp"
#include "MNIST_loader.hpp"

using cnn::Network;

template <typename T>
arma::Mat<T> flatten(std::vector<arma::Col<T>> const&);
Network::Mat flatten_images(std::vector<loader::Image> const&);
Network::VecList labels_to_outputs(std::vector<loader::Label> const&);
Network::VecList images_to_inputs(std::vector<loader::Image> const&);
loader::Label output_to_label(Network::Vec const& output);

int main()
{
    std::vector<loader::Image> train_images, test_images;
    std::vector<loader::Label> train_labels, test_labels;

    std::cerr << "Loading MNIST dataset..." << std::endl;
    std::tie(train_images, train_labels, test_images, test_labels)
        = loader::load_MNIST_data();

    std::cerr << "Flattening images" << std::endl;
    auto flattened_images = flatten(images_to_inputs(train_images));
    auto test_inputs = images_to_inputs(test_images);
    std::cerr << "Flattening labels" << std::endl;
    auto flattened_labels = flatten(labels_to_outputs(train_labels));
    std::cerr << "Done." << std::endl;

    // train a network on MNIST with stochastic gradient descent
    const unsigned epochs = 1;
    const unsigned batch_size = 10;
    const unsigned n_batches = flattened_images.n_cols / batch_size;


    Network mnist_fc({784, 30, 10});

    std::cerr << "Training FC net with MNIST data" << std::endl;
    for (unsigned epoch = 0; epoch < epochs; ++epoch) {
        arma::shuffle(flattened_images, 1);

        /*
        for (unsigned batch = 0; batch < n_batches; ++batch) {
            const unsigned long long start_index = batch * batch_size;
            const unsigned long long end_index =
                std::min(start_index + batch_size, flattened_images.n_cols - 1);

            mnist_fc.train(
                flattened_images.cols(start_index, end_index),
                flattened_labels.cols(start_index, end_index),
                0.1);
        }
        */
        for (unsigned i = 0; i < 5; ++i) {
            const auto start_cost =
                cnn::Network::cost(flattened_labels.col(i),
                                   mnist_fc.classify(flattened_images.col(i)));
            auto x = mnist_fc.classify(flattened_images.col(i));
            mnist_fc.train(flattened_images.col(i), flattened_labels.col(i), 100);
            const auto end_cost =
                cnn::Network::cost(flattened_labels.col(i),
                                   mnist_fc.classify(flattened_images.col(i)));
            std::cout << i << ": " << start_cost << " -> " << end_cost << std::endl;
            Network::Mat actual_pred(flattened_labels.col(i));
            actual_pred.insert_cols(1,x);
            std::cout << "Training\n" << actual_pred << std::endl;

        }
        //mnist_fc.train(flattened_images, flattened_labels, 0.1);
        auto test_outputs = mnist_fc.classify(test_inputs);
        int correct = std::inner_product(
            test_outputs.begin(), test_outputs.end(), test_labels.begin(), 0,
            std::plus<int>{},
            [] (Network::Vec const& output, loader::Label y) -> int {
                auto x = output_to_label(output);
                //std::cout << int(x) << ' ' << int(y) << std::endl;
                return x == y;
            });
        std::cerr << "Completed epoch " << epoch + 1 << "/" << epochs << std::endl;
        std::cerr << correct << " correct" << std::endl;
    }

    /*
    // train a simple network on a single input
    Network network({2,7,4});
    for (int i = 0; i < 100; ++i) {
        network.train(Network::Mat(Network::Vec{1,1}),
                      Network::Mat(Network::Vec{1,1,0,1}), 0.5);
        const auto cost = cnn::Network::cost({1,1,0,1}, network.classify(Network::Vec{1,1}));
        std::cout << i << ", " << cost << std::endl;
        std::cout << network.classify(Network::Vec{1,1}) << std::endl;
    }
    */
}

template <typename T>
arma::Mat<T> flatten(std::vector<arma::Col<T>> const& list)
{
    arma::Mat<T> flattened(list[0].n_elem, list.size());
    for (size_t i = 0; i < list.size(); ++i) {
        flattened.col(i) = list[i];
    }
    return flattened;
}

Network::VecList images_to_inputs(std::vector<loader::Image> const& l) {
    Network::VecList columnated;
    std::transform(l.begin(), l.end(), std::back_inserter(columnated),
           [] (loader::Image const& image) -> Network::Vec {
                 return arma::vectorise(arma::conv_to<Network::Mat>::from(image)) / 255;
             });
    return columnated;
}

Network::Vec label_to_output(loader::Label scalar_label)
{
    // it is necessary to convert from a scalar [0,9] to a vector representing
    // the desired output of the neural network, which is a 10-element zero
    // vector with a single 1 at the label index.
    Network::Vec output_label(10, arma::fill::zeros);
    output_label[scalar_label] = 1;
    return output_label;
}

loader::Label output_to_label(Network::Vec const& output) {
    arma::uword label;
    output.max(label);
    //std::cout << output << std::endl;
    //std::cout << label << std::endl;
    return label;
}

Network::VecList labels_to_outputs(std::vector<loader::Label> const& labels) {
    std::vector<Network::Vec> outputs;
    outputs.reserve(labels.size());
    std::transform(labels.begin(), labels.end(), std::back_inserter(outputs),
                   label_to_output);
    return outputs;
}
