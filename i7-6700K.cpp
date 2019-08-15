#include <iostream>
#include <random>
#include <chrono>
#include <immintrin.h>

#define malloc_m256(n) (__m256 *)_mm_malloc(sizeof(__m256) * n, 64);

using namespace std;

void runLoop(bool print) {
	for (int exponent = 6; exponent < 24; exponent ++) {
		int SIZE = 2 << exponent;
		// The main arrays that will be multiplied.
		__m256 * mul01 = malloc_m256(SIZE);
		__m256 * mul02 = malloc_m256(SIZE);
		__m256 * res   = malloc_m256(SIZE);

		// Initialize random number generation.
		time_t seed = time(nullptr);
		default_random_engine engine(seed);
		uniform_real_distribution<float> distribution(0.0, 1.0);

		// Temporary stack for storing the random numbers as they are generated.
		float tmp01[8] __attribute__((aligned(32)));
		float tmp02[8] __attribute__((aligned(32)));

		// cout << "Beginning random number generation." << endl;

		// Time the random number generation process.
		auto begin = chrono::high_resolution_clock::now();
		for (int i = 0; i < SIZE; i++) {
			for (int j = 0; j < 8; j++) {
				tmp01[j] = distribution(engine);
				tmp02[j] = distribution(engine);
			}
			mul01[i] = _mm256_load_ps((float const *)tmp01);
			mul02[i] = _mm256_load_ps((float const *)tmp02);
		}

		// Display the amount of time that it took.
		auto end      = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
		// cout << "Generated " << SIZE * 2 * 8 << " random single precision floating point numbers." << endl;
		// cout << "Process took " << duration << "ns" << endl;

		// cout << "Beginning multiplication." << endl;

		// Run the loop a few times to give the CPU enough info
		// to perform caching and other tasks.

		for (int k = 0; k < 1024; k++) {
			for (int i = 0; i < SIZE; i++) {
				res[i] = _mm256_mul_ps(mul01[i], mul02[i]);
			}
		}
		

		begin = chrono::high_resolution_clock::now();
		for (int i = 0; i < SIZE; i++) {
			res[i] = _mm256_mul_ps(mul01[i], mul02[i]);
		}
		end      = chrono::high_resolution_clock::now();
		duration = chrono::duration_cast<chrono::nanoseconds>(end - begin).count();
		// cout << "Multiplication took " << duration << "ns" << endl;

		long flops = (SIZE * 8) / (duration / 1e9);
		// cout << "Multiplications = " << SIZE * 8         << endl;
		// cout << "Duration        = " << (duration / 1e9) << "s" << endl;
		// cout << "Single core floating point operations per second = " << flops << endl;

		// cout << "First 10 Results: " << endl;

		if (print) {
			cout << SIZE * 8 << " " << duration << endl;
		}
		
		float in01[8]   __attribute__((aligned(32)));
		float in02[8]   __attribute__((aligned(32)));
		float restmp[8] __attribute__((aligned(32)));
		for (int i = 0; i < 10; i ++) {

			_mm256_store_ps(in01, mul01[i]);
			_mm256_store_ps(in02, mul02[i]);
			_mm256_store_ps(restmp, res[i]);

			// cout << "\tresult " << i << endl;
			for (int j = 0; j < 8; j++) {
				// cout << "\t\t" << in01[j] << " * " << in02[j] << " = " << restmp[j] << endl;
			}
		}

		_mm_free(mul01);
		_mm_free(mul02);
		_mm_free(res);
	}
}

int main(int argc, char **argv) {

	for (int i = 0; i < 10; i++) {
		runLoop(false);
	}
	
	runLoop(true);

	return 0;
}