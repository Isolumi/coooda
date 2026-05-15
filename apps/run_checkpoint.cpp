#include <coooda_cpp/ops/vector.hpp>

#include <iostream>

int main() {
    const auto out = coooda_cpp::ops::vector_add_reference({1.0f}, {2.0f});
    std::cout << "Checkpoint runner ready. cpp vector smoke result=" << out[0] << "\n";
    return 0;
}
