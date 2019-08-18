#include <iostream>
#include <random>
#include <chrono>
#include <immintrin.h>
#include <thread>
#include <vector>

#define malloc_m512(n) (__m512 *)_mm_malloc(sizeof(__m512) * n, 64);
#define malloc_m256(n) (__m256 *)_mm_malloc(sizeof(__m256) * n, 64);

using namespace std;

int avx512_test(int *t_start, int *t_stop, int warmup, int idx, int size, int rep) {
	// Initialize the random number generator.
	time_t seed = time(nullptr);
	default_random_engine engine(seed);
	uniform_real_distribution<float> distribution(0.0, 1.0);

	// Determine how large to make the arrays to operate on.
	// The two arrays should fit in the L1, with at least 
	// 1KB left over.

	// The first division by two is because there are two threads
	// per core. The second is because there are two arrays to use.
	int bytes_per_array   = (((size - 2048) / 2) / 2);
	int vectors_per_array = bytes_per_array   / 32;
	int singles_per_array = vectors_per_array * 8;

	// The results will be stored in place, in the mul01 array.
	__m256 * mul01 = malloc_m256(vectors_per_array);
	__m256 * mul02 = malloc_m256(vectors_per_array);

	// cout << "THREAD " << idx << " :: ";
	// cout << "# of single precision values allocated = ";
	// cout << singles_per_array * 2 << endl;

	// Temporary stack for the floats that we generate.
	float tmp01[8] __attribute__((aligned(64)));
	float tmp02[8] __attribute__((aligned(64)));

	// Fill the arrays with random values in [0, 1]
	for (int i = 0; i < vectors_per_array; ++i) {
		for (int j = 0; j < 8; ++j) {
			tmp01[j] = distribution(engine);
			tmp02[j] = distribution(engine);
		}
		mul01[i] = _mm256_load_ps((const float *)tmp01);
		mul02[i] = _mm256_load_ps((const float *)tmp02);
	}

	// Run the multiplication loops so that the CPU can warmup.
	for (int i = 0; i < warmup; ++i) {
		for (int j = 0; j < vectors_per_array; ++j) {
			mul01[j] = _mm256_mul_ps(mul01[j], mul02[j]);
		}
	}

	// Figure out the start time and write it to the assigned address.
	*t_start = chrono::high_resolution_clock::now().time_since_epoch().count();

	__m256 result = mul01[0];
	// Run the calculation.
	for (int r = 0; r < rep*vectors_per_array; ++r) {
			// mul01[j] = _mm512_mul_ps(mul01[j], mul02[j]);
			result = _mm256_fmadd_ps(result, result, result);
	}

	*t_stop   = chrono::high_resolution_clock::now().time_since_epoch().count();

	mul01[0] = result;

	// Sum the result of the calculation so that the compiler won't optimize
	// the whole loop away.
	float sum = 0.0;
	for (int j = 0; j < vectors_per_array; ++j) {
		_mm256_store_ps(tmp01, mul01[j]);

		for (int i = 0; i < 8; ++i) {
			sum += tmp01[i];
		}
	}

	// Print the total.
	// cout << "THREAD " << idx << " :: ";
	// cout << "Total = " << sum << endl;

	// cout << "THREAD " << idx << " :: ";
	// cout << "N FLOPS = " << singles_per_array*rep << endl;

	// Return the number of flops.
	return 2 * singles_per_array * rep;
}

int ____avx512_test(int *t_start, int *t_stop, int warmup, int idx, int size, int rep) {
	// Initialize the random number generator.
	time_t seed = time(nullptr);
	default_random_engine engine(seed);
	uniform_real_distribution<float> distribution(0.0, 1.0);

	// Determine how large to make the arrays to operate on.
	// The two arrays should fit in the L1, with at least 
	// 1KB left over.

	// The first division by two is because there are two threads
	// per core. The second is because there are two arrays to use.
	int bytes_per_array   = (((size - 2048) / 2) / 2);
	int vectors_per_array = bytes_per_array   / 32;
	int singles_per_array = vectors_per_array * 8;

	// The results will be stored in place, in the mul01 array.
	__m256 * mul01 = malloc_m256(vectors_per_array);
	__m256 * mul02 = malloc_m256(vectors_per_array);

	// cout << "THREAD " << idx << " :: ";
	// cout << "# of single precision values allocated = ";
	// cout << singles_per_array * 2 << endl;

	// Temporary stack for the floats that we generate.
	float tmp01[8] __attribute__((aligned(64)));
	float tmp02[8] __attribute__((aligned(64)));

	// Fill the arrays with random values in [0, 1]
	for (int i = 0; i < vectors_per_array; ++i) {
		for (int j = 0; j < 8; ++j) {
			tmp01[j] = distribution(engine);
			tmp02[j] = distribution(engine);
		}
		mul01[i] = _mm256_load_ps((const float *)tmp01);
		mul02[i] = _mm256_load_ps((const float *)tmp02);
	}

	// Run the multiplication loops so that the CPU can warmup.
	for (int i = 0; i < warmup; ++i) {
		for (int j = 0; j < vectors_per_array; ++j) {
			mul01[j] = _mm256_mul_ps(mul01[j], mul02[j]);
		}
	}

	// Figure out the start time and write it to the assigned address.
	*t_start = chrono::high_resolution_clock::now().time_since_epoch().count();

	// Run the calculation.
	for (int r = 0; r < rep; ++r) {
		for (int j = 0; j < vectors_per_array; ++j) {
			// mul01[j] = _mm512_mul_ps(mul01[j], mul02[j]);
			mul01[j] = _mm256_fmadd_ps(mul01[j], mul01[j], mul01[j]);
		}
	}

	*t_stop   = chrono::high_resolution_clock::now().time_since_epoch().count();

	// Sum the result of the calculation so that the compiler won't optimize
	// the whole loop away.
	float sum = 0.0;
	for (int j = 0; j < vectors_per_array; ++j) {
		_mm256_store_ps(tmp01, mul01[j]);

		for (int i = 0; i < 8; ++i) {
			sum += tmp01[i];
		}
	}

	// Print the total.
	// cout << "THREAD " << idx << " :: ";
	// cout << "Total = " << sum << endl;

	// cout << "THREAD " << idx << " :: ";
	// cout << "N FLOPS = " << singles_per_array*rep << endl;

	// Return the number of flops.
	return 2 * singles_per_array * rep;
}

// start  :: Start time of the actual processing for the thread, used to
//          calculate the total time.
// end    :: End time of the actual processing, used to calculate total time.
// warmup :: How many times to run the calculation without timing, to get the
//           CPU into the proper state for calculations. 
// idx    :: Thread index, used to display messages.
// size   :: The size, in bytes, of the cache to run the test for. 
// 
// This function is designed to be run on multiple threads at a time. The idea
// is that each thread will write out its start and end times so that the 
// controlling process can determine the total execution time.
int __avx512_test(int *t_start, int *t_stop, int warmup, int idx, int size, int rep) {
	// Initialize the random number generator.
	time_t seed = time(nullptr);
	default_random_engine engine(seed);
	uniform_real_distribution<float> distribution(0.0, 1.0);

	// Determine how large to make the arrays to operate on.
	// The two arrays should fit in the L1, with at least 
	// 1KB left over.

	// The first division by two is because there are two threads
	// per core. The second is because there are two arrays to use.
	int bytes_per_array   = (((size - 2048) / 2) / 2);
	int vectors_per_array = bytes_per_array   / 64;
	int singles_per_array = vectors_per_array * 16;

	// The results will be stored in place, in the mul01 array.
	__m512 * mul01 = malloc_m512(vectors_per_array);
	__m512 * mul02 = malloc_m512(vectors_per_array);

	// cout << "THREAD " << idx << " :: ";
	// cout << "# of single precision values allocated = ";
	// cout << singles_per_array * 2 << endl;

	// Temporary stack for the floats that we generate.
	float tmp01[16] __attribute__((aligned(64)));
	float tmp02[16] __attribute__((aligned(64)));

	// Fill the arrays with random values in [0, 1]
	for (int i = 0; i < vectors_per_array; ++i) {
		for (int j = 0; j < 16; ++j) {
			tmp01[j] = distribution(engine);
			tmp02[j] = distribution(engine);
		}
		mul01[i] = _mm512_load_ps((void const *)tmp01);
		mul02[i] = _mm512_load_ps((void const *)tmp02);
	}

	// Run the multiplication loops so that the CPU can warmup.
	for (int i = 0; i < warmup; ++i) {
		for (int j = 0; j < vectors_per_array; ++j) {
			mul01[j] = _mm512_mul_ps(mul01[j], mul02[j]);
		}
	}

	// Figure out the start time and write it to the assigned address.
	*t_start = chrono::high_resolution_clock::now().time_since_epoch().count();

	// Run the calculation.
	for (int r = 0; r < rep; ++r) {
		for (int j = 0; j < vectors_per_array; ++j) {
			mul01[j] = _mm512_mul_ps(mul01[j], mul02[j]);
		}
	}

	*t_stop   = chrono::high_resolution_clock::now().time_since_epoch().count();

	// Sum the result of the calculation so that the compiler won't optimize
	// the whole loop away.
	float sum = 0.0;
	for (int j = 0; j < vectors_per_array; ++j) {
		_mm512_store_ps(tmp01, mul01[j]);

		for (int i = 0; i < 16; ++i) {
			sum += tmp01[i];
		}
	}

	// Print the total.
	// cout << "THREAD " << idx << " :: ";
	// cout << "Total = " << sum << endl;

	// cout << "THREAD " << idx << " :: ";
	// cout << "N FLOPS = " << singles_per_array*rep << endl;

	// Return the number of flops.
	return singles_per_array * rep;
}

void mean_time(int iterations, int idx, float *mean_result, int size, int rep, int warmup) {
	int L1_PER_CORE = 32*1024;

	int start;
	int stop;

	float mean  = 0.0;
	int   flops = 0; 

	for (int i = 0; i < iterations; ++i) {
		// Run single threaded, sized for the L1 cache.
		flops = avx512_test(&start, &stop, warmup, idx, size, rep);
		int duration = stop - start;
		mean += (float)duration;
	}

	mean = mean / iterations;

	*mean_result = mean;
}

int main(int argc, char **argv) {
	int L1_PER_CORE = 32*1024;
	int L2_PER_CORE = 1024*1024;
	int L3_PER_CORE = 1408*1024;
	int N_CORES     = 16;
	int N_THREADS   = 32;
	int REP         = 128;
	int WARMUP      = 128*1;

	int average_over = 1024;

	int start;
	int stop;

	float mean  = 0.0;
	int   flops = 0;

	for (int i = 0; i < average_over; ++i) {
		// Run single threaded, sized for the L1 cache.
		flops = avx512_test(&start, &stop, WARMUP, 0, L1_PER_CORE, REP);
		int duration = stop - start;
		mean += (float)duration;
	}

	mean = mean / average_over;

	cout << "Mean Duration      = " << mean << "ns" << endl;
	cout << "Single core GFlops = " << (flops / (mean / 1e9)) / 1e9 << endl;

	// Now we perform the same operation, on 32 threads.

	thread *threads = new thread[N_THREADS];
	float  *means   = new float[N_THREADS];
	for (int t = 0; t < N_THREADS; ++t) {
		threads[t] = thread(
			mean_time, 
			average_over, 
			t, 
			&means[t], 
			L1_PER_CORE, 
			REP, 
			WARMUP
		);
	}

	for (int t = 0; t < N_THREADS; ++t) {
		threads[t].join();
	}

	// Calculate the number of flops for each code.
	float *gflops = new float[N_THREADS];
	float total   = 0.0;
	for (int t = 0; t < N_THREADS; ++t) {
		gflops[t] = (flops / (means[t] / 1e9)) / 1e9;
		total += gflops[t];
	}

	cout << "Total multicore GFLOPS = " << total << endl;
	cout << "Per Thread GFLOPS :: " << endl;
	for (int t = 0; t < N_THREADS; ++t) {
		cout << "\t" << t << " " << gflops[t] << endl;
	}


	return 0;
}