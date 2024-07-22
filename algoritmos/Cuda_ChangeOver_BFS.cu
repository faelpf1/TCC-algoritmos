#include <bits/stdc++.h>
#include <cstdlib>
#include <sdsl/bit_vectors.hpp>

using namespace std;

void init(int argc, char* argv[], int& numRep, int& K, int& threadsNumber)
{
    if(argc < 4) 
    {
        cout << "Put all parameters: 1-File 2-K value 3-ThreadNumber" << '\n';
        exit(1);
    }
    else if(argc < 5) numRep = 1;
    else numRep = atoi(argv[4]);

    K = atoi(argv[2]);
    if(K<2)
    {
        cout << "Put K >= 2" << '\n';
        exit(1);
    }

    threadsNumber = atoi(argv[3]);
    if(threadsNumber > 1024)
    {
        cout << "Put threadsNumber <= 1024";
        exit(1);
    }  
}

void fileOpenErrorCheck(ifstream& file)
{
    if(!file) 
    {
        cout << "Failed to open the file" << '\n';
        exit(1);
    }
}

class BitVectorMatrix {
private:
    u_int64_t bSize;
    size_t bTrueSize;
public:
    u_int64_t* matrix_h;
    u_int64_t n_physical, n_virtual;
    BitVectorMatrix(u_int64_t n_physical, u_int64_t n_virtual)
    {
        this->n_physical = n_physical;
        this->n_virtual = n_virtual;
        bSize = n_physical*n_physical;
        bTrueSize = ceil((double)bSize/64);
        matrix_h = (u_int64_t*) malloc(sizeof(u_int64_t) * bTrueSize);  
        memset(matrix_h, 0, bTrueSize); 
    }
    ~BitVectorMatrix()
    {
        free(matrix_h);
    }

    void set(u_int64_t index, bool bit) 
    {
        if(bit) matrix_h[index >> 6]  |= (1ULL << (index & 63)); /* set 1 */
        else matrix_h[index >> 6] &= ~(1ULL << (index & 63)); /* set 0 */
    }

    void setBit(int i, int j)
    {
        u_int64_t index = i * (u_int64_t)n_physical + j;
        set(index, 1);
    }
   
    int get(int i, int j)
    {
        u_int64_t index = i * (u_int64_t)n_physical + j;
        return ( 1 & ( matrix_h[index >> 6] >> (index & 63) ) );
    }
    
    // void print_matrix() //colocar para imprimir a matriz virtual
    // {
    //     for(size_t i = 0; i < n_vertex; i++) 
    //     {
    //         for(size_t j = 0; j < n_vertex; j++) 
    //         {
    //             cout << get(i, j) << '\t';
    //         }
    //         cout << '\n';
    //     }
    // }
};

struct k2_tree_node
{
    int i, j, level;
    k2_tree_node() = default;
    k2_tree_node(int i, int j, int level) : i(i), j(j), level(level) { }
};

struct submatrix_coordinate
{
    int min_i, min_j, max_i, max_j;
};

__device__ int getToDevice(u_int64_t* b, u_int64_t n_physical, int i, int j)
{
    u_int64_t index = i * n_physical + j;
    return 1 & (b[index >> 6] >> (index & 63));
}

__device__ bool change_Over_BFS_loop(u_int64_t* matrix_d, u_int64_t n_physical, int min_i, int min_j, int max_i, int max_j)
{
    int th_number = blockDim.x;
    int th_idx = threadIdx.x + min_j;
    for(int i = min_i; i < max_i; i++)
    {
        if(i >= n_physical) return 0;
        for(int j = th_idx; j < max_j; j+=th_number)
        {
            if(j >= n_physical) break;
            if(getToDevice(matrix_d, n_physical, i, j) == 1) return 1;
        }
    }
    return 0;
}

__global__ void change_Over_BFS_Parallel(u_int64_t* matrix_d, u_int64_t n_physical, submatrix_coordinate* submatrix_d, int* found_d)
{  
    int blk_idx = blockIdx.x; 
    int min_i = submatrix_d[blk_idx].min_i, min_j = submatrix_d[blk_idx].min_j;
    int max_i = submatrix_d[blk_idx].max_i, max_j = submatrix_d[blk_idx].max_j;
    
    int f = change_Over_BFS_loop(matrix_d, n_physical, min_i, min_j, max_i, max_j);
    if(f) found_d[blk_idx] = 1;    
}

int change_over_BFS_sequential(BitVectorMatrix& matrix, submatrix_coordinate submatrix_h)
{
    u_int64_t n_physical = matrix.n_physical;
    int min_i = submatrix_h.min_i, min_j = submatrix_h.min_j;
    int max_i = submatrix_h.max_i, max_j = submatrix_h.max_j;
    for (int i = min_i; i < max_i; i++)
    {
        if(i >= n_physical) return 0;
        for (int j = min_j; j < max_j; j++)
        {
            if(j >= n_physical) break;
            else if((matrix.get(i, j) == 1)) return 1;
        }
    }
    return 0;
}

void leaf_level_case(BitVectorMatrix& matrix, sdsl::bit_vector& k_l, size_t& l, deque<k2_tree_node>& d)
{
    k2_tree_node submatrix = d.front();
    u_int64_t n_physical = matrix.n_physical;
    if((submatrix.i < n_physical) && (submatrix.j < n_physical)) k_l[l] = matrix.get(submatrix.i, submatrix.j);
    else k_l[l] = 0;
    l++;
    d.pop_front();
}

void get_submatrix_nodes(deque<k2_tree_node>& d, u_int64_t n_virtual, int K, int KPOWER)
{
    k2_tree_node submatrix = d.front();
    /* Calculate the size of each child submatrix */
    int child_size = n_virtual / (1 << int(KPOWER * ( submatrix.level + 1 )) ); 
    /* Performs the divisions in K ^ 2 submatrices */
    for(int i = 0; i < K; i++)
    {
        for(int j = 0; j < K; j++)
        {
            k2_tree_node node;
            node.i = submatrix.i + i * child_size;
            node.j = submatrix.j + j * child_size;
            node.level = submatrix.level + 1;
            d.push_back(node);
        }
    }
}

void submatrix_min_max(submatrix_coordinate* submatrix_h, deque<k2_tree_node>& d, int matrix_size)
{
    int idx = 0;
    for(auto coord = d.begin(); coord != d.end(); ++coord)
    {
        submatrix_h[idx].min_i = coord->i;
        submatrix_h[idx].min_j = coord->j;
        submatrix_h[idx].max_i = coord->i + matrix_size;
        submatrix_h[idx].max_j = coord->j + matrix_size;
        idx++;
    }
}

void copy_to_bitvector_T_and_divide_submatrix(int found, sdsl::bit_vector& k_t, size_t& t, deque<k2_tree_node>& d, u_int64_t n_virtual, int K, int KPOWER)
{
    k_t[t] = found;
    t++;
    if(found == 1) get_submatrix_nodes(d, n_virtual, K, KPOWER);
    d.pop_front();
}

void tree_level_case(u_int64_t* matrix_d, BitVectorMatrix& matrix, deque<k2_tree_node>& d, sdsl::bit_vector& k_t, size_t& t, int matrix_size, int K, int KPOWER, int th_number)
{
    u_int64_t n_virtual = matrix.n_virtual;
    u_int64_t n_physical = matrix.n_physical;
    int d_size = d.size();
    
    /* Calculate min and max coordinates from k2_tree current level */
    int submatrix_coordinate_bytes = sizeof(submatrix_coordinate) * d_size;
    submatrix_coordinate* submatrix_h = (submatrix_coordinate*) malloc(submatrix_coordinate_bytes);
    submatrix_min_max(submatrix_h, d, matrix_size);
    
    int found_bytes = sizeof(int) * d_size;
    int* found_h = (int*) malloc(found_bytes);

    if(matrix_size >= 32)
    {   
        /* Allocate and copy coordinates to device */
        submatrix_coordinate* submatrix_d;
        cudaMalloc(&submatrix_d, submatrix_coordinate_bytes);
        cudaMemcpy(submatrix_d, submatrix_h, submatrix_coordinate_bytes, cudaMemcpyHostToDevice);
        free(submatrix_h);
        
        /* Allocate and init found to device */
        int* found_d;
        cudaMalloc(&found_d, found_bytes);
        cudaMemset(found_d, 0, found_bytes);
        int threadsPerBlock = ((matrix_size >> 10) > 0) ? th_number : matrix_size;
        int blocksPerGrid = d_size;
 
        change_Over_BFS_Parallel<<<blocksPerGrid, threadsPerBlock>>>(matrix_d, n_physical, submatrix_d, found_d);

        cudaMemcpy(found_h, found_d, found_bytes, cudaMemcpyDeviceToHost);
        cudaFree(submatrix_d);
        cudaFree(found_d);

        for(int i = 0; i < d_size; i++)
        {
            copy_to_bitvector_T_and_divide_submatrix(found_h[i], k_t, t, d, n_virtual, K, KPOWER);
        }
    }
    else
    {
        for(int i = 0; i < d_size; i++)
        {
            found_h[i] = change_over_BFS_sequential(matrix, submatrix_h[i]);
            copy_to_bitvector_T_and_divide_submatrix(found_h[i], k_t, t, d, n_virtual, K, KPOWER);
        }  
        free(submatrix_h);
    }
    free(found_h);
}

void root_level_case(deque<k2_tree_node>& d, u_int64_t n_virtual, int K, int KPOWER)
{
    get_submatrix_nodes(d, n_virtual, K, KPOWER);
    d.pop_front();
}

void write_Bitvector(sdsl::bit_vector& k_t, sdsl::bit_vector& k_l, size_t t, size_t l)
{
    /* Write results in files */
    ofstream K_T("./outputs/KT_CUDA.sdsl"), K_L("./outputs/KL_CUDA.sdsl");
    k_t.resize(t);
    k_l.resize(l);
    sdsl::serialize(k_t, K_T);
    sdsl::serialize(k_l, K_L);
}

void changeOver_BFS(BitVectorMatrix& matrix, int K, int KPOWER, int th_number)
{
    u_int64_t n_virtual = matrix.n_virtual;
    u_int64_t n_physical = matrix.n_physical;
    /* Create and allocate values from host matrix to device matrix */
    u_int64_t *matrix_d;
    u_int64_t b_trueSize_bytes = ceil((double)(n_physical*n_physical/64)) * sizeof(u_int64_t);
    cudaMalloc(&matrix_d, b_trueSize_bytes);
    cudaMemcpy(matrix_d, matrix.matrix_h, b_trueSize_bytes, cudaMemcpyHostToDevice);  
    
    /* Deque of submatrix orign coordinates */
    deque<k2_tree_node> d;
    d.push_back({0, 0, 0}); /* Inserts the matrix starting at level 0 with root */

    /* Create k2_tree bitvector and index */
    u_int64_t vet_size = n_virtual * n_virtual;
    sdsl::bit_vector k_t = sdsl::bit_vector(vet_size, 0), k_l = sdsl::bit_vector(vet_size, 0);
    size_t t = 0, l = 0; /* Bitvectors index */

    while(!d.empty())
    {
        int matrix_size = n_virtual / ( 1 << int( KPOWER * (d.front().level) ) );
        if(matrix_size == 1) leaf_level_case(matrix, k_l, l, d);
        else if(matrix_size == 0) break; /* Empty matrix case */
        else if(matrix_size != n_virtual) tree_level_case(matrix_d, matrix, d, k_t, t, matrix_size, K, KPOWER, th_number);
        else if(matrix_size == n_virtual) root_level_case(d, n_virtual, K, KPOWER); 
    }
    cudaFree(matrix_d);
    write_Bitvector(k_t, k_l, t, l);
}

template <typename T>
double time_diff(const T& start, const T& stop)
{
    chrono::duration<double> time = stop - start;
    return time.count();
}

template <typename Fun>
double average_time(Fun&& func, BitVectorMatrix& matrix, int K, int KPOWER, int threadsNumber, int numRep)
{
    double sum = 0;
    for (int r = 0; r != numRep; r++)
    {
        auto t1 = chrono::steady_clock::now();
        func(matrix, K, KPOWER, threadsNumber);
        auto t2 = chrono::steady_clock::now();
        sum += time_diff(t1, t2);
    }
    return sum / numRep;
}

int main(int argc, char* argv[])
{
    int numRep, K, KPOWER, threadsNumber, i, j;
    float logk_n;    
    u_int64_t n_vertex_physical, n_vertex_virtual;
    init(argc, argv, numRep, K, threadsNumber);

    ifstream file(argv[1]);
    fileOpenErrorCheck(file);
    file >> n_vertex_physical;
    
    /* Maintains original size or calculates approximate size with K^|log K n| */
    n_vertex_virtual = n_vertex_physical;
    KPOWER = log2(K);
    logk_n = log(n_vertex_physical) / log(K);
    if( !( logk_n == (int)(logk_n) ) ) n_vertex_virtual = 1 << (int)(KPOWER * ceil(logk_n)); 

    cout << "# Vertex number: " << n_vertex_virtual << " Matrix size: " << n_vertex_virtual * n_vertex_virtual << '\n';
    
    BitVectorMatrix matrix(n_vertex_physical, n_vertex_virtual);
    while (file >> i && file >> j) matrix.setBit(i-1, j-1); /* File index begins in 1 */
    file.close();  

    /* Benchmark time calculation module */
    auto time = average_time(changeOver_BFS, matrix, K, KPOWER, threadsNumber, numRep);  
    cout << "Build from Matrix Parallel time = " << time << "s\n";
    return 0;
}