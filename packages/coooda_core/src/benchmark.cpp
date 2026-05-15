#include <coooda_core/benchmark.hpp>

#include <chrono>
#include <exception>
#include <iostream>

namespace {

void print_available_cases(std::ostream &out, const std::vector<coooda_core::bench::BenchmarkCase> &cases) {
    out << "available cases:\n";
    for (const coooda_core::bench::BenchmarkCase &benchmark_case : cases) {
        out << "  " << benchmark_case.name << " - " << benchmark_case.description << "\n";
    }
}

void print_usage(
    std::ostream &out,
    const char *program_name,
    const std::vector<coooda_core::bench::BenchmarkCase> &cases
) {
    out << "usage: " << program_name << " [--list] [--case <name>]\n";
    print_available_cases(out, cases);
}

const coooda_core::bench::BenchmarkCase *find_case(
    const std::vector<coooda_core::bench::BenchmarkCase> &cases,
    const std::string &name
) {
    for (const coooda_core::bench::BenchmarkCase &benchmark_case : cases) {
        if (benchmark_case.name == name) {
            return &benchmark_case;
        }
    }
    return nullptr;
}

bool run_case(const coooda_core::bench::BenchmarkCase &benchmark_case) {
    try {
        benchmark_case.run();
        return true;
    } catch (const std::exception &error) {
        std::cerr << benchmark_case.name << " failed: " << error.what() << "\n";
    } catch (...) {
        std::cerr << benchmark_case.name << " failed with an unknown error\n";
    }
    return false;
}

} // namespace

namespace coooda_core::bench {

BenchmarkResult time_once(const std::string &name, const std::function<void()> &fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = stop - start;
    return BenchmarkResult{name, elapsed.count()};
}

void print_result(const BenchmarkResult &result) {
    std::cout << result.name << ": " << result.elapsed_ms << " ms\n";
}

int run_cases(int argc, char **argv, const std::vector<BenchmarkCase> &cases) {
    const char *program_name = argc > 0 ? argv[0] : "benchmark";

    if (cases.empty()) {
        std::cerr << "no benchmark cases registered\n";
        return 1;
    }

    if (argc == 1) {
        bool ok = true;
        for (const BenchmarkCase &benchmark_case : cases) {
            ok = run_case(benchmark_case) && ok;
        }
        return ok ? 0 : 1;
    }

    if (argc == 2 && std::string(argv[1]) == "--list") {
        print_available_cases(std::cout, cases);
        return 0;
    }

    if (argc == 3 && std::string(argv[1]) == "--case") {
        const BenchmarkCase *benchmark_case = find_case(cases, argv[2]);
        if (benchmark_case == nullptr) {
            std::cerr << "unknown benchmark case: " << argv[2] << "\n";
            print_usage(std::cerr, program_name, cases);
            return 1;
        }
        return run_case(*benchmark_case) ? 0 : 1;
    }

    std::cerr << "invalid benchmark arguments\n";
    print_usage(std::cerr, program_name, cases);
    return 1;
}

} // namespace coooda_core::bench
