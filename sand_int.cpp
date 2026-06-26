#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
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

    std::vector<int> H;
    std::vector<std::mt19937> rngs;
    
    static constexpr double tanTheta = 0.803847577293368; //3.0 * std::tan(15.0 * M_PI / 180.0)

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

        return py::array(py::dtype::of<int>(), {Ny, Nx},
            {static_cast<ssize_t>(Nx * sizeof(int)), static_cast<ssize_t>(-sizeof(int))},
            H.data() + (Nx - 1));
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

                int val = n + dist(gen);
                H[i] = val;
            }
        }
    }

    // ===== SHADOW =====
    inline bool isShadow(int x, int y) const {
        int i0 = idx(x, y);
        int h0 = H[i0];

        for (int dx = 1; dx < shadowCheckDistance; ++dx) {

            int xx = x - dx;  // upstream (wind from left)
            int h  = H[idx(xx, y)];

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

            for (int k = 0; k < 4; ++k) {
                auto [dx,dy] = order[k];

                int xn = x + dx;
                int yn = y + dy;
                int j = idx(xn, yn);

                int diff = H[i] - H[j];
                if (diff > avalancheThreshold) {
                    H[i] -= 1;
                    H[j] += 1;
                    stack.emplace_back(x,y);
                    stack.emplace_back(xn,yn);
                    break;
                }
                else if (diff < -avalancheThreshold) {
                    H[j] -= 1;
                    H[i] += 1;
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
        if (H[i] <= 0) return;
        if (isShadow(x,y)) return;

        H[i] -= 1;
        relaxAvalanche(x, y, gen);

        for (int step = 0; step < Nx; ++step) {
            x += hopLength;
            int j = idx(x,y);
            bool shadow = isShadow(x,y);
            double p = shadow ? 1.0 : (H[j] > 0 ? 0.6 : 0.4);
            if (prob(gen) < p) {
                H[j] += 1;
                relaxAvalanche(x, y, gen);
                break;
            }
        }
    }

    int blockcount = 4;
    // ===== FULL STEP =====
    void step() {
        for (int i_start = 0; i_start < blockcount; ++i_start) {
            for (int j_start = 0; j_start < blockcount; ++j_start) {
                #pragma omp parallel for collapse(2)
                for (int bi = i_start; bi < NBx; bi += blockcount) {
                    for (int bj = j_start; bj < NBy; bj += blockcount) {
                            auto& gen = rngs[bi * NBy + bj];
                            stepBlock(bi, bj, gen);
                    }
                }
            }
        }
    }
};


PYBIND11_MODULE(dune_int, m) {
    m.doc() = "Dune field simulation";

    py::class_<DuneField>(m, "DuneField")
        .def(py::init<int,int,int,int,int>(), py::arg("nbx"), py::arg("nby"),
             py::arg("bx"), py::arg("by"), py::arg("init_height") = 3)

        .def("runNsteps", &DuneField::runNsteps, py::arg("N_steps"));
}