#include "MNIST_loader.hpp"

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

/*
std::basic_istream<char>& operator>>(std::basic_istream<char>& is, arma::Mat<unsigned char>& im) {
    return is;
}*/

namespace loader {

using Data = std::tuple<std::vector<Image>,  // training images
                        std::vector<Label>,  // training labels
                        std::vector<Image>,  // testing images
                        std::vector<Label>>; // testing labels

Label label_from_number(byte scalar_label);
std::vector<Image> load_image_set(std::string const& path);
std::vector<Label> load_label_set(std::string const& path);

Data load_MNIST_data()
{
    const std::string train_image_path = "data/train-images-idx3-ubyte.dat";
    const std::string train_label_path = "data/train-labels-idx1-ubyte.dat";
    const std::string test_image_path  = "data/t10k-images-idx3-ubyte.dat";
    const std::string test_label_path  = "data/t10k-labels-idx1-ubyte.dat";

    return Data(
        load_image_set(train_image_path),
        load_label_set(train_label_path),
        load_image_set(test_image_path),
        load_label_set(test_label_path)
    );

}

std::vector<Label> load_label_set(std::string const& path)
{
    using Iter = std::istream_iterator<byte>;

    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (not file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    // validate file
    int32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    if (magic_number != 0x801) {
        std::stringstream error;
        error << "Label file has wrong magic number: " << magic_number;
        throw std::runtime_error(error.str());
    }

    // read number of data, preallocate array
    int32_t n_labels;
    file.read(reinterpret_cast<char*>(&n_labels), 4);
    std::vector<Label> labels(n_labels);

    // copy data
    std::transform(Iter(file), Iter(), labels.begin(), label_from_number);
                
    return labels;
}

std::vector<Image> load_image_set(std::string const& path)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (not file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    int magic_number, n_images, image_x, image_y;
    file.read(reinterpret_cast<char*>(&magic_number), 4);

    if (magic_number != 0x803) {
        std::stringstream error;
        error << "Image file has wrong magic number: " << magic_number;
        throw std::runtime_error(error.str());
    }

    file.read(reinterpret_cast<char*>(&n_images), 4);
    file.read(reinterpret_cast<char*>(&image_x), 4);
    file.read(reinterpret_cast<char*>(&image_y), 4);

    std::vector<Image> images(n_images, Image(image_x, image_y));
    arma::Mat<byte> flipped_identity({0, 0, 0, 1,
                                      0, 0, 1, 0,
                                      0, 1, 0, 0,
                                      1, 0, 0, 0});
    flipped_identity.reshape(4, 4);
    for (auto& im : images) {
        file.read(reinterpret_cast<char*>(im.memptr()), image_x * image_y);
        im = im.t();

        // MNIST is broken, so this is needed to fix it
        // Every 4 columns are flipped in x
        for (int col = 0; col < image_x; col += 4) {
            im.cols(col, col+3) = im.cols(col, col+3) * flipped_identity;
        }
    }

    return images;
}

Label label_from_number(byte scalar_label)
{
    // it is necessary to convert from a scalar [0,9] to a vector representing
    // the desired output of the neural network, which is a 10-element zero
    // vector with a single 1 at the label index.
    Label output_label(10, arma::fill::zeros);
    output_label[scalar_label] = 1;
    return output_label;
}
}
