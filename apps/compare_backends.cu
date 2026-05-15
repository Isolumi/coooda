#include <coooda_compare/ops/vector_compare.hpp>

#include <iostream>

int main() {
    const bool ok = coooda_compare::ops::vector_add_matches_reference();
    std::cout << "Backend comparison smoke: " << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
