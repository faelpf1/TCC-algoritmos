#include <bits/stdc++.h>
#include <cstdlib>
#include <omp.h>
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
    u_int64_t* b;
    u_int64_t bSize;
    u_int64_t n_vertex;
    size_t bTrueSize;

    void initWithZeros()
    {
        for(size_t i = 0; i < bTrueSize; i++)
        {
            b[i]=0;
        }
    }
public:
    BitVectorMatrix(u_int64_t size)
    {
        n_vertex = size;
        bSize = size*size;
        bTrueSize = ceil((double)bSize/64);
        b = (u_int64_t*) malloc(sizeof(u_int64_t) * bTrueSize);  
        initWithZeros();
    }
    ~BitVectorMatrix()
    {
        free(b);
    }

    void set(u_int64_t index, bool bit) 
    {
        if(bit) b[index >> 6]  |= (1ULL << (index & 63)); /* set 1 */
        else b[index >> 6] &= ~(1ULL << (index & 63)); /* set 0 */
    }
   
    int getBit(u_int64_t index)
    {
        return 1 & (b[index >> 6] >> (index & 63));
    }
    
    u_int64_t size() 
    {
        return bSize;
    }

    int get(int i, int j)
    {
        return getBit(i * n_vertex + j);
    }
  
    void linkBitLinearMatrix(int i, int j)
    {
        set(i * n_vertex + j, 1);
    }

    void print_matrix() 
    {
        for(size_t i = 0; i < n_vertex; i++) 
        {
            for(size_t j = 0; j < n_vertex; j++) 
            {
                cout << get(i, j) << '\t';
            }
            cout << '\n';
        }
    }
};

struct k2_tree_node
{
    int i, j, level;

    k2_tree_node() = default;
    k2_tree_node(int i, int j, int level) :
        i(i), j(j), level(level) { }
};

template <typename T>
double time_diff(const T& start, const T& stop)
{
    chrono::duration<double> time = stop - start;
    return time.count();
}

template <typename Fun>
double average_time(Fun&& func, BitVectorMatrix& matrix, u_int64_t n, int K, int KPOWER, int t, int numRep)
{
    double sum = 0;

    for (int r = 0; r != numRep; r++)
    {
        auto t1 = chrono::steady_clock::now();
        func(matrix, n, K, KPOWER, t);
        auto t2 = chrono::steady_clock::now();
        sum += time_diff(t1, t2);
    }
    return sum / numRep;
}

int build_from_matrix_helper_sequential(BitVectorMatrix& mat, k2_tree_node& matrix, int n)
{
    const auto max_size_i = matrix.i+n, max_size_j = matrix.j+n;
    const auto min_size_i = matrix.i, min_size_j = matrix.j;

    int found = 0;
    for (int i = min_size_i; i < max_size_i; i+=2)
    {
        for (int j = min_size_j; j < max_size_j; j+=2)
        {
            if (mat.get(i, j) == 1)
            {
                found = 1;
                break;
            }
            else if (mat.get(i+1, j) == 1)
            {
                found = 1;
                break;
            }
            else if (mat.get(i, j+1) == 1)
            {
                found = 1;
                break;
            }
            else if (mat.get(i+1, j+1) == 1)
            {
                found = 1;
                break;
            }
        }
        if (found)
        {
            break;
        }
    }
    return found;
}

int build_from_matrix_helper_parallel(BitVectorMatrix& mat, k2_tree_node& matrix, int n)
{
    const auto max_size_i = matrix.i+n, max_size_j = matrix.j+n;
    const auto min_size_i = matrix.i, min_size_j = matrix.j;

    int found = 0;

//cout << "# Does OpenMP allow cancel construct: " << omp_get_cancellation() << endl;

    #pragma omp parallel default(none) shared(min_size_i, min_size_j, max_size_i, max_size_j, mat, found)
    {
        #pragma omp for
        for (int i = min_size_i; i < max_size_i; i+=2)
        {
            for (int j = min_size_j; j < max_size_j; j+=2)
            {
                if (mat.get(i, j) == 1)
                {
                    #pragma omp critical
                    found = 1;
                    #pragma omp cancel for
                }
                else if (mat.get(i+1, j) == 1)
                {
                    #pragma omp critical
                    found = 1;
                    #pragma omp cancel for
                }
                else if (mat.get(i, j+1) == 1)
                {
                    #pragma omp critical
                    found = 1;
                    #pragma omp cancel for
                }
                else if (mat.get(i+1, j+1) == 1)
                {
                    #pragma omp critical
                    found = 1;
                    #pragma omp cancel for
                }
            }
            #pragma omp cancellation point for
        }
    }
    return found;
}

void build_from_matrix(BitVectorMatrix& mat, u_int64_t size, int K, int power, int thrds)
{
    queue<k2_tree_node> q;
    q.push({0, 0, 0}); // Inserts the matrix starting at (0,0) which is at level 0
    sdsl::bit_vector k_t, k_l;
    u_int64_t vet_size = size * size;
    k_t = sdsl::bit_vector(vet_size, 0);
    k_l = sdsl::bit_vector(vet_size, 0);
    size_t t = 0, l = 0;
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(thrds); // Use thrds threads for all consecutive parallel regions

    while (!q.empty())
    {
        auto matrix = q.front(); // Get matrix
        auto rt = -1, rl = -1;
        auto matrix_size = size / (1 << power * (matrix.level));

        if(matrix_size == 1) // Leaf level base case
        {
            rl = mat.get(matrix.i, matrix.j);
            k_l[l] = rl;
            l++;
        }
        else if(matrix_size == 0)
            break;
        else if (size == matrix_size) ;
        else
        {
            if(matrix_size <= 64)
                rt = build_from_matrix_helper_sequential(mat,matrix,matrix_size);
            else
                rt = build_from_matrix_helper_parallel(mat,matrix,matrix_size);
            k_t[t] = rt;
            t++;
        }
        q.pop();

        if(rt==0) {
            continue;    // The matrix only has zeros, there is no need to process the children
        }

        int child_size = size / (1 << power * (matrix.level + 1)); // Calculate the size of each child submatrix
        /* Performs the divisions in K ^ 2 submatrices */
        for(int i=0; i<K; i++)
        {
            for(int j=0; j<K; j++)
            {
                k2_tree_node node;
                node.i = matrix.i + i * child_size;
                node.j = matrix.j + j* child_size;
                node.level = matrix.level+1;
                q.push(node);
            }
        }
    }
    ofstream K_T ("./outputs/KT5.sdsl"), K_L ("./outputs/KL5.sdsl");
    k_t.resize(t);
    k_l.resize(l);

    sdsl::serialize(k_t, K_T);
    sdsl::serialize(k_l, K_L);
}

int main(int argc, char* argv[])
{
    int numRep, K, KPOWER, t, i, j;
    float logk_n;
    u_int64_t n_vertex;

    init(argc, argv, numRep, K, t);

    ifstream file(argv[1]);
    fileOpenErrorCheck(file);
    file >> n_vertex;
    
    KPOWER = log2(K);
    logk_n = log(n_vertex) / log (K); /* For testing based on final arity */
    if(!(logk_n == (int)logk_n)) n_vertex = 1 << (int)(KPOWER * ceil(logk_n)); /* Maintains original size or calculates approximate size with K^|log K n| */
     
    cout << "# Vertex Number = " << n_vertex << " Matrix Size = " << n_vertex * n_vertex << '\n';

    BitVectorMatrix matrix(n_vertex);
    while(file >> i && file >> j) matrix.linkBitLinearMatrix(i-1, j-1); /* File index begins in 1 */
    file.close();

    /* Benchmark time calculation module */
    auto time = average_time(build_from_matrix, matrix, n_vertex, K, KPOWER, t, numRep);

    cout << "Build from Matrix Parallel time = " << time << "s\n";

    return 0;
}
