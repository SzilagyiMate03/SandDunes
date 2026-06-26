#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <random>
#include <filesystem>

namespace py = pybind11;

// ---------- DuneFieldGPU CLASS ----------
class DuneFieldGPU {
public:
    //------- SIMULATION PARAMETERS -------
    int Bx, By, NBx, NBy, Nx, Ny;
    int hopLength = 5;
    int shadowCheckDist;
    int blockcount = 4;   // chessboard step count (same as CPU)

    //------- OPENCL OBJECTS -------
    cl::Context       context;
    cl::CommandQueue  queue;
    cl::Program       program;
    cl::Kernel        kernelStep;
    cl::Buffer        bufH;        // H[Nx*Ny]  (int)
    cl::Buffer        bufRng;      // rng[NBx*NBy] (uint)

    std::vector<int>  H_host;      // CPU-side copy (for reading back)

    //------- CONSTRUCTOR -------
    DuneFieldGPU(int nbx, int nby, int bx, int by, int init_height = 3,
                 const std::string& kernelPath = "dune_gpu.cl",
                 int platformIdx = 0, int deviceIdx   = 0)
        : Bx(bx), By(by), NBx(nbx), NBy(nby),
          Nx(nbx*bx), Ny(nby*by),
          H_host(nbx*bx * nby*by)
    {
        shadowCheckDist = int(1.5f * std::sqrt(float(Nx * init_height)));

        // ------ OPENCL PLATFORM + DEVICE SELECTION ------
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
            throw std::runtime_error("Nincs elerheto OpenCL platform!");
        if (platformIdx >= (int)platforms.size())
            throw std::runtime_error("Hibas platform index");

        cl::Platform plat = platforms[platformIdx];

        std::vector<cl::Device> devices;
        plat.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (devices.empty())
            throw std::runtime_error("Nincs elerheto OpenCL eszkoz!");
        if (deviceIdx >= (int)devices.size())
            throw std::runtime_error("Hibas device index");

        cl::Device dev = devices[deviceIdx];

        py::print("[DuneGPU] Platform:", plat.getInfo<CL_PLATFORM_NAME>());
        py::print("[DuneGPU] Device  :", dev.getInfo<CL_DEVICE_NAME>());

        //------ CONTEXT AND QUEUE ------
        context = cl::Context(dev);
        queue   = cl::CommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE);

        // ------ KERNEL ------
        std::string src = loadKernelSource(kernelPath);
        program = cl::Program(context, src);
        try {
            program.build({dev}, "-cl-fast-relaxed-math");
        } catch (const cl::Error& e) {
            std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
            throw std::runtime_error(
                std::string("Kernel build hiba:\n") + log);
        }
        kernelStep = cl::Kernel(program, "stepBlocks");

        // ------ BUFFER ALLOCATION ------
        bufH   = cl::Buffer(context, CL_MEM_READ_WRITE, Nx * Ny * sizeof(int));
        bufRng = cl::Buffer(context, CL_MEM_READ_WRITE, NBx * NBy * sizeof(cl_uint));

        // ------ INITIALIZATION ------
        initH(init_height);
        initRng();
    }

    // ------ RUN N STEPS -----
    py::array_t<int> runNsteps(int N_steps) {
        for (int s = 0; s < N_steps; ++s) {
            step();
        }

        // GPU → CPU copy
        queue.enqueueReadBuffer(bufH, CL_TRUE, 0, Nx * Ny * sizeof(int), H_host.data());

        // Return as NumPy array (same orientation as CPU version)
        return py::array(
            py::dtype::of<int>(), {Ny, Nx},
            {static_cast<py::ssize_t>(Nx * sizeof(int)),
             static_cast<py::ssize_t>(-sizeof(int))},
            H_host.data() + (Nx - 1));
    }

    // ------ LISTS AVAILABLE DEVICES ------
    static std::string listDevices() {
        std::ostringstream out;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        for (int pi = 0; pi < (int)platforms.size(); ++pi) {
            out << "Platform " << pi << ": "
                << platforms[pi].getInfo<CL_PLATFORM_NAME>() << "\n";
            std::vector<cl::Device> devs;
            platforms[pi].getDevices(CL_DEVICE_TYPE_ALL, &devs);
            for (int di = 0; di < (int)devs.size(); ++di) {
                out << "  Device " << di << ": "
                    << devs[di].getInfo<CL_DEVICE_NAME>() << "\n";
            }
        }
        return out.str();
    }

private:
    //---------- LOAD KERNEL SOURCE ----------
    static std::string loadKernelSource(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error(
                "Nem sikerult megnyitni a kernel forrasfajlt: " + path);
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    // ----- INITIALIZE H, COPY TO GPU -----
    void initH(int n) {
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(-1, 1);
        for (auto& v : H_host) v = n + dist(gen);

        queue.enqueueWriteBuffer(bufH, CL_TRUE, 0, H_host.size() * sizeof(int), H_host.data());
    }

    // ---- INITIALIZE RNG STATES, COPY TO GPU -----
    void initRng() {
        int nBlocks = NBx * NBy;
        std::vector<cl_uint> rngInit(nBlocks);
        for (int bi = 0; bi < NBx; ++bi)
            for (int bj = 0; bj < NBy; ++bj) {
                cl_uint seed = (cl_uint)((bi+1) * 73856093u ^ (bj+1) * 19349663u);
                rngInit[bi * NBy + bj] = seed;
            }
        queue.enqueueWriteBuffer(bufRng, CL_TRUE, 0, nBlocks * sizeof(cl_uint), rngInit.data());
    }

    // ----RUNS A FULL STEP (CHESSBOARD REFRESH) -----
    void step() {
        for (int i_start = 0; i_start < blockcount; ++i_start) {
            for (int j_start = 0; j_start < blockcount; ++j_start) {

                // Number of active blocks in this chessboard phase
                int activeI = (NBx - i_start + blockcount - 1) / blockcount;
                int activeJ = (NBy - j_start + blockcount - 1) / blockcount;
                if (activeI <= 0 || activeJ <= 0) continue;

                // Kernel arguments
                int arg = 0;
                kernelStep.setArg(arg++, bufH);
                kernelStep.setArg(arg++, bufRng);
                kernelStep.setArg(arg++, (cl_int)Nx);
                kernelStep.setArg(arg++, (cl_int)Ny);
                kernelStep.setArg(arg++, (cl_int)Bx);
                kernelStep.setArg(arg++, (cl_int)By);
                kernelStep.setArg(arg++, (cl_int)hopLength);
                kernelStep.setArg(arg++, (cl_int)shadowCheckDist);
                kernelStep.setArg(arg++, (cl_int)blockcount);
                kernelStep.setArg(arg++, (cl_int)i_start);
                kernelStep.setArg(arg++, (cl_int)j_start);

                // Setting up the global and local work sizes
                cl::NDRange global(activeI, activeJ);
                cl::NDRange local(1, 1);   // one work-item = one blokk

                // Enqueue the kernel for execution
                queue.enqueueNDRangeKernel(kernelStep, cl::NullRange, global, local);
                queue.finish();  // barrier between chessboard phases
            }
        }
    }
};

//--------- PYBIND11 MODUL ---------
PYBIND11_MODULE(dune_gpu, m) {
    m.def("list_devices", &DuneFieldGPU::listDevices);

    py::class_<DuneFieldGPU>(m, "DuneFieldGPU")
        .def(py::init<int,int,int,int,int,const std::string&,int,int>(),
             py::arg("nbx"), py::arg("nby"), py::arg("bx"), py::arg("by"),
             py::arg("init_height") = 3, py::arg("kernel_path") = "dune_gpu.cl",
             py::arg("platform_idx") = 0, py::arg("device_idx")   = 0)

        .def("runNsteps", &DuneFieldGPU::runNsteps, py::arg("N_steps"))

        .def_readwrite("hop_length", &DuneFieldGPU::hopLength);
}
