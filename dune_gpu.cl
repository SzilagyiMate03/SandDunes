__constant float TAN_THETA = 0.803847577f;

__constant char perms[24][4][2] = {
        {{ 1,0}, {-1,0}, { 0,1}, { 0,-1}}, {{ 1,0}, {-1,0}, { 0,-1}, { 0,1}},
        {{ 1,0}, { 0,1}, {-1,0}, { 0,-1}}, {{ 1,0}, { 0,1}, { 0,-1}, {-1,0}},
        {{ 1,0}, { 0,-1}, {-1,0}, { 0,1}}, {{ 1,0}, { 0,-1}, { 0,1}, {-1,0}},
        {{-1,0}, { 1,0}, { 0,1}, { 0,-1}}, {{-1,0}, { 1,0}, { 0,-1}, { 0,1}},
        {{-1,0}, { 0,1}, { 1,0}, { 0,-1}}, {{-1,0}, { 0,1}, { 0,-1}, { 1,0}},
        {{-1,0}, { 0,-1}, { 1,0}, { 0,1}}, {{-1,0}, { 0,-1}, { 0,1}, { 1,0}},
        {{ 0,1}, { 1,0}, {-1,0}, { 0,-1}}, {{ 0,1}, { 1,0}, { 0,-1}, {-1,0}},
        {{ 0,1}, {-1,0}, { 1,0}, { 0,-1}}, {{ 0,1}, {-1,0}, { 0,-1}, { 1,0}},
        {{ 0,1}, { 0,-1}, { 1,0}, {-1,0}}, {{ 0,1}, { 0,-1}, {-1,0}, { 1,0}},
        {{ 0,-1}, { 1,0}, {-1,0}, { 0,1}}, {{ 0,-1}, { 1,0}, { 0,1}, {-1,0}},
        {{ 0,-1}, {-1,0}, { 1,0}, { 0,1}}, {{ 0,-1}, {-1,0}, { 0,1}, { 1,0}},
        {{ 0,-1}, { 0,1}, { 1,0}, {-1,0}}, {{ 0,-1}, { 0,1}, {-1,0}, { 1,0}}
    };

#define AVALANCHE_MAX_ITER 1000
#define STACK_SIZE 256

// ---------- CELL INDEX ----------
inline int cellIdx(int x, int y, int Nx, int Ny){
    x = (x + Nx) % Nx;
    y = (y + Ny) % Ny;

    return y * Nx + (Nx - 1 - x);
}

// ---------- RNG ----------
inline uint xorshift32(uint* state){
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// ---------- SHADOW CHECK ----------
inline bool isShadow(__global const int* H, int x, int y, int Nx, int Ny, int shadowDist){
    int h0 = H[cellIdx(x, y, Nx, Ny)];

    for (int dx = 1; dx < shadowDist; ++dx) {
        int xx = x - dx;
        int h  = H[cellIdx(xx, y, Nx, Ny)];

        if ((h - h0) > TAN_THETA * (dx - 0.5f)) {
            return true;
        }
    }
    return false;
}

//---------- AVALANCHE ----------
inline void relaxAvalanche(__global int* H, int x0, int y0, int Nx, int Ny, uint* rngState){
    int stackX[STACK_SIZE];
    int stackY[STACK_SIZE];

    int top = 0; // denotes the current size of the stack

    stackX[top] = x0;
    stackY[top] = y0;
    top++;

    int iter = 0;
    while (top > 0 && iter < AVALANCHE_MAX_ITER) {
        iter++;
        top--; 

        int x = stackX[top];
        int y = stackY[top];

        int p = (int)(xorshift32(rngState) % 24u);

        int i_idx= cellIdx(x, y, Nx, Ny);
        int h_i = H[i_idx];

        for (int k = 0; k < 4; ++k) {

            int dx = perms[p][k][0];
            int dy = perms[p][k][1];

            int xn = x + dx;
            int yn = y + dy;

            int j_idx = cellIdx(xn, yn, Nx, Ny);
            int h_j = H[j_idx];

            int diff = h_i - h_j;
            if (diff > 2) {
                H[i_idx] = h_i - 1;
                H[j_idx] = h_j + 1;
                if (top + 2 < STACK_SIZE) {
                    stackX[top] = x;
                    stackY[top] = y;
                    top++;

                    stackX[top] = xn;
                    stackY[top] = yn;
                    top++;
                }
                break;
            }
            else if (diff < -2) {
                H[j_idx] = h_j - 1;
                H[i_idx] = h_i + 1;
                if (top + 2 < STACK_SIZE) {
                    stackX[top] = x;
                    stackY[top] = y;
                    top++;

                    stackX[top] = xn;
                    stackY[top] = yn;
                    top++;
                }
                break;
            }
        }
    }
}

// ---------- KERNEL: 1 BLOCK STEP = 1 WORK ITEM ----------
__kernel void stepBlocks(__global int* H, __global uint* rngStates, int Nx, int Ny, int Bx, int By,
                         int hopLength, int shadowCheckDistance, int blockcount, int i_start, int j_start){

    const int NBx = Nx / Bx;
    const int NBy = Ny / By;

    int bi = i_start + get_global_id(0) * blockcount;
    int bj = j_start + get_global_id(1) * blockcount;

    if (bi >= NBx || bj >= NBy) return;

    uint rng = rngStates[bi * NBy + bj]; // local copy of RNG state

    int lx = (int)(xorshift32(&rng) % (uint)Bx);
    int ly = (int)(xorshift32(&rng) % (uint)By);

    int x = bi * Bx + lx;
    int y = bj * By + ly;

    int i = cellIdx(x, y, Nx, Ny);

    if (H[i] <= 0) {
        rngStates[bi * NBy + bj] = rng; // save back the RNG state
        return;
    }

    if (isShadow(H, x, y, Nx, Ny, shadowCheckDistance)) {
        rngStates[bi * NBy + bj] = rng;
        return;
    }

    H[i] -= 1;
    relaxAvalanche(H, x, y, Nx, Ny, &rng);

    for (int s = 0; s < Nx; ++s) {
        x += hopLength;

        int j = cellIdx(x, y, Nx, Ny);
        bool shadow = isShadow(H, x, y, Nx, Ny, shadowCheckDistance);
        float u = (float)(xorshift32(&rng) >> 8) * (1.0f / 16777216.0f); //random number in [0,1)

        float p = shadow ? 1.0f : (H[j] > 0 ? 0.6f : 0.4f);

        if (u < p) {
            H[j] += 1;
            relaxAvalanche(H, x, y, Nx, Ny, &rng);
            break;
        }
    }
    rngStates[bi * NBy + bj] = rng;
}