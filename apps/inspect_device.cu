#include <coooda_core/device.hpp>

#include <exception>
#include <iostream>

int main() {
    try {
        std::cout << coooda_core::cuda::device_summary() << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cout << "CUDA device inspection skipped: " << error.what() << "\n";
        return 0;
    }
}
