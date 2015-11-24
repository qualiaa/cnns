#include "Network.hpp"

int main()
{
    cnn::Network network({2,50,4});

    network.train({1,1}, {1,1,0,1});
}
