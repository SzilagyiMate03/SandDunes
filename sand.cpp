#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <atomic>
#include <random>
#include <array>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

class DuneField {
public:
    // ===== PARAMÉTEREK =====
    int Bx, By;     // blokk méret
    int NBx, NBy;   // blokkok száma
    int Nx, Ny;     // teljes grid

    int hopLength = 5;
    int avalancheThreshold = 2;
    int shadowCheckDistance;

    std::vector<std::atomic<int>> H;
    std::vector<std::mt19937> rngs;
    
    static constexpr double tanTheta = 3.0 * std::tan(15.0 * M_PI / 180.0);

    static constexpr std::array<std::array<std::pair<int,int>,4>,24> perms = {{
        {{{ 1,0}, {-1,0}, { 0,1}, { 0,-1}}}, {{{ 1,0}, {-1,0}, { 0,-1}, { 0,1}}},
        {{{ 1,0}, { 0,1}, {-1,0}, { 0,-1}}}, {{{ 1,0}, { 0,1}, { 0,-1}, {-1,0}}},
        {{{ 1,0}, { 0,-1}, {-1,0}, { 0,1}}}, {{{ 1,0}, { 0,-1}, { 0,1}, {-1,0}}},
        {{{-1,0}, { 1,0}, { 0,1}, { 0,-1}}}, {{{-1,0}, { 1,0}, { 0,-1}, { 0,1}}},
        {{{-1,0}, { 0,1}, { 1,0}, { 0,-1}}}, {{{-1,0}, { 0,1}, { 0,-1}, { 1,0}}},
        {{{-1,0}, { 0,-1}, { 1,0}, { 0,1}}}, {{{-1,0}, { 0,-1}, { 0,1}, { 1,0}}},
        {{{ 0,1}, { 1,0}, {-1,0}, { 0,-1}}}, {{{ 0,1}, { 1,0}, { 0,-1}, {-1,0}}},
        {{{ 0,1}, {-1,0}, { 1,0}, { 0,-1}}}, {{{ 0,1}, {-1,0}, { 0,-1}, { 1,0}}},
        {{{ 0,1}, { 0,-1}, { 1,0}, {-1,0}}}, {{{ 0,1}, { 0,-1}, {-1,0}, { 1,0}}},
        {{{ 0,-1}, { 1,0}, {-1,0}, { 0,1}}}, {{{ 0,-1}, { 1,0}, { 0,1}, {-1,0}}},
        {{{ 0,-1}, {-1,0}, { 1,0}, { 0,1}}}, {{{ 0,-1}, {-1,0}, { 0,1}, { 1,0}}},
        {{{ 0,-1}, { 0,1}, { 1,0}, {-1,0}}}, {{{ 0,-1}, { 0,1}, {-1,0}, { 1,0}}}
    }};

    // ===== KONSTRUKTOR =====
    DuneField(int nbx, int nby, int bx, int by, int n = 3)
        : Bx(bx), By(by),
        NBx(nbx), NBy(nby),
        Nx(nbx * bx),
        Ny(nby * by),
        H(Nx * Ny),
        rngs(NBx * NBy)
    {
        //RNG init
        for (int bi = 0; bi < NBx; ++bi) {
            for (int bj = 0; bj < NBy; ++bj) {

                unsigned seed = bi * 73856093 ^ bj * 19349663;
                rngs[bi * NBy + bj] = std::mt19937(seed);
            }
        }

        // height grid init
        initialize(n);
        // calculate shadow check distance based on max expected height difference
        shadowCheckDistance = int(1.5 * std::sqrt(Nx * n));
    }
    
    // ===== PYTHON STEP =====
    py::array_t<int> runNsteps(int N_steps) {

        for (int s = 0; s < N_steps; ++s) {
            step();
        }

        py::array_t<int> arr({Ny, Nx});
        auto buf = arr.mutable_unchecked<2>();

        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                buf(y,x) = H[idx(x,y)].load(std::memory_order_relaxed);
            }
        }

        return arr;
    }

private:
    // ===== INDEXELÉS =====
    inline int idx(int x, int y) const {
        x = (x + Nx) % Nx;
        y = (y + Ny) % Ny;
        return y * Nx + Nx - 1 - x;
    }

    // ===== INITIALIZATION =====
    void initialize(int n) {
        #pragma omp parallel
        {
            std::mt19937 gen(42 + omp_get_thread_num());
            std::uniform_int_distribution<int> dist(-1,1);

            #pragma omp for
            for (int i = 0; i < Nx * Ny; ++i) {

                int val = n + dist(gen);  // n vagy n+1
                H[i].store(val, std::memory_order_relaxed);
            }
        }
    }

    // ===== SHADOW =====
    inline bool isShadow(int x, int y) const {
        int i0 = idx(x, y);
        int h0 = H[i0].load(std::memory_order_relaxed);

        for (int dx = 1; dx < shadowCheckDistance; ++dx) {

            int xx = x - dx;  // upstream (wind from left)
            int h  = H[idx(xx, y)].load(std::memory_order_relaxed);

            if (((h - h0)) > tanTheta * (dx - 0.5)) {
                return true;
            }
        }
        return false;
    }

    // ===== AVALANCHE =====
    inline void relaxAvalanche(int x0, int y0, std::mt19937& gen) {
        std::vector<std::pair<int,int>> stack;
        stack.emplace_back(x0, y0);

        while (!stack.empty()) {
            auto [x,y] = stack.back();
            stack.pop_back();

            int p = gen() % 24;
            const auto& order = perms[p];

            int i = idx(x,y);
            int hi = H[i].load(std::memory_order_relaxed);

            for (int k = 0; k < 4; ++k) {
                auto [dx,dy] = order[k];

                int xn = x + dx;
                int yn = y + dy;
                int j = idx(xn, yn);
                int hj = H[j].load(std::memory_order_relaxed);

                int diff = hi - hj;
                if (diff > avalancheThreshold) {
                    H[i].fetch_sub(1, std::memory_order_relaxed);
                    H[j].fetch_add(1, std::memory_order_relaxed);
                    stack.emplace_back(x,y);
                    stack.emplace_back(xn,yn);
                    break;
                }
                else if (diff < -avalancheThreshold) {
                    H[j].fetch_sub(1, std::memory_order_relaxed);
                    H[i].fetch_add(1, std::memory_order_relaxed);
                    stack.emplace_back(x,y);
                    stack.emplace_back(xn,yn);
                    break;
                }
            }
        }
    }

    // ===== BLOCK STEP =====
    void stepBlock(int bi, int bj, std::mt19937& gen) {

        std::uniform_real_distribution<> prob(0.0,1.0);

        int x = bi*Bx + gen() % Bx;
        int y = bj*By + gen() % By;
        int i = idx(x,y);
        int hi = H[i].load(std::memory_order_relaxed);
        if (hi <= 0) return;
        if (isShadow(x,y)) return;

        H[i].fetch_sub(1, std::memory_order_relaxed);
        relaxAvalanche(x, y, gen);

        int maxSteps = Nx;
        for (int step = 0; step < maxSteps; ++step) {
            x += hopLength;
            int j = idx(x,y);
            int hj = H[j].load(std::memory_order_relaxed);
            bool shadow = isShadow(x,y);
            double p = shadow ? 1.0 : (hj > 0 ? 0.6 : 0.4);
            if (prob(gen) < p) {
                H[j].fetch_add(1, std::memory_order_relaxed);
                relaxAvalanche(x, y, gen);
                break;
            }
        }
    }

    // ===== FULL STEP =====
    void step() {
        for (int i_start = 0; i_start < 2; ++i_start) {
            for (int j_start = 0; j_start < 2; ++j_start) {
                #pragma omp parallel for collapse(2)
                for (int bi = i_start; bi < NBx; bi += 2) {
                    for (int bj = j_start; bj < NBy; bj += 2) {
                            auto& gen = rngs[bi * NBy + bj];
                            stepBlock(bi, bj, gen);
                    }
                }
            }
        }
    }
};


PYBIND11_MODULE(dune, m) {
    m.doc() = "Dune field simulation";

    py::class_<DuneField>(m, "DuneField")
        .def(py::init<int,int,int,int,int>(), py::arg("nbx"), py::arg("nby"),
             py::arg("bx"), py::arg("by"), py::arg("shadowCheckDistance") = 3)

        .def("runNsteps", &DuneField::runNsteps, py::arg("N_steps"));
}