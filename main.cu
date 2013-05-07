#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <complex>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>
#include <map>
#include <stdexcept>
#include <cuda.h>
#include <cufft.h>

//#include <helper_functions.h>
//#include <helper_cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA_ERROR:\ncode:%s\nfile: %s\nline:%d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////
////
////    Constants & Typedefs
////
///////////////////////////////////////////////////////////////////////////////////////////////

#define PI 3.14159265359
#define BLOCK_SIZE 4096
#define SIGNAL_THRESHOLD 200
#define MAX_TRANSMISSIONS 200

//172MHz gives us CB
#define SAMPLE_RATE 172089331.259
#define BATCH_SIZE 1
#define NUM_STREAMS 2
#define BLOCKS_PER_STREAM 16

#define HzInMHz 1000000

typedef char byte;
typedef float2 Complex;

///////////////////////////////////////////////////////////////////////////////////////////////
////
////    Device Variables
////
///////////////////////////////////////////////////////////////////////////////////////////////

__device__ int transmissionCount;

__device__ int timeStep;

///////////////////////////////////////////////////////////////////////////////////////////////
////
////    Kernels
////
///////////////////////////////////////////////////////////////////////////////////////////////

void __device__ createTransmission( int idx ,
		int* transmissionBins,
		cufftReal* scaledResultBuffer,
		cufftReal* transmissionFrequencies,
		cufftReal* transmissionStarts,
		cufftReal* transmissionStrengths,
		bool* activeTransmissions
	)
{

	transmissionBins[ transmissionCount - 1 ] = idx;

	//frequency in MHz
	transmissionFrequencies[ transmissionCount - 1 ] = idx * SAMPLE_RATE / BLOCK_SIZE / HzInMHz;

	transmissionStarts[ transmissionCount - 1 ] = timeStep / SAMPLE_RATE;

	transmissionStrengths[ transmissionCount - 1 ] = scaledResultBuffer[ idx ];

	activeTransmissions[ idx ] = true;

}

void __device__ finishTransmission( int idx,
		int* transmissionBins,
		cufftReal* transmissionEnds,
		bool* activeTransmissions
	)
{

	for( int i = transmissionCount - 1 ; i >= 0 ; i-- )
	{

		if( transmissionBins[ i ] == idx )
		{

			transmissionEnds[ i ] = timeStep / SAMPLE_RATE;

			activeTransmissions[ idx ] = false;

			return;

		}

	}

}

void __global__ scaleResult( cufftReal* scaledResultBuffer , cufftComplex* resultBuffer )
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if( idx < BLOCK_SIZE )
	{

		scaledResultBuffer[ idx ] = sqrt( resultBuffer[ idx ].x * resultBuffer[ idx ].x * +
										  resultBuffer[ idx ].y * resultBuffer[ idx ].y      );

		scaledResultBuffer[ idx ] = 20 * log10( scaledResultBuffer[ idx ] );

	}

}

void __global__ initTransmissionArray( bool* activeTransmissions )
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	transmissionCount = 0;

	if( idx < BLOCK_SIZE )

		activeTransmissions[ idx ] = false;

}

void __global__ findTransmissions(
		cufftReal* scaledResultBuffer ,
		int* deviceBins,
		cufftReal *deviceFrequencies,
		cufftReal *deviceStarts,
		cufftReal *deviceEnds,
		cufftReal *deviceStrengths,
		bool* activeTransmissions
	)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if( idx < BLOCK_SIZE && idx != 0 )
	{

		if( scaledResultBuffer[ idx ] > SIGNAL_THRESHOLD && activeTransmissions[ idx ] == false )
		{

			atomicAdd( &transmissionCount , 1 );

			createTransmission( idx , deviceBins , scaledResultBuffer , deviceFrequencies , deviceStarts , deviceStrengths, activeTransmissions );

		}

		if( scaledResultBuffer[ idx ] < SIGNAL_THRESHOLD && activeTransmissions[ idx ] == true )
		{

			finishTransmission( idx , deviceBins , deviceEnds , activeTransmissions );

		}

	}

	//timeStep += BLOCK_SIZE;

	atomicAdd( &timeStep , 1 );

}

///////////////////////////////////////////////////////////////////////////////////////////////
////
////    Device Pointers
////
///////////////////////////////////////////////////////////////////////////////////////////////

cufftComplex *deviceResult = 0;

cufftReal *deviceSource = 0;

cufftReal *deviceScaledResult = 0;

int* deviceBins = 0;

cufftReal *deviceFrequencies = 0;

cufftReal *deviceStarts = 0;

cufftReal *deviceEnds = 0;

cufftReal *deviceStrengths = 0;

bool* deviceActiveTransmissions = 0;

int* deviceCount = 0;

///////////////////////////////////////////////////////////////////////////////////////////////
////
////    Host Variables
////
///////////////////////////////////////////////////////////////////////////////////////////////

int* hostBins;

cufftReal *hostFrequencies;

cufftReal *hostStarts;

cufftReal *hostEnds;

cufftReal *hostStrengths;

///////////////////////////////////////////////////////////////////////////////////////////////
////
////    Functions
////
///////////////////////////////////////////////////////////////////////////////////////////////

int main( int argc , char** argv )
{
	
	std::string filename = std::string( argv[1] );

	std::ifstream f;

	struct stat filestatus;

	stat( filename.c_str() , &filestatus );

	size_t filesize = filestatus.st_size;

	f.open( filename.c_str() , std::ios::in | std::ios::binary );

	if( !f.good() )
	{
	
		std::cerr << "Can't open file" << std::endl;

		exit( 1 );

	}

	cufftReal* original = 0;

	//std::cout << "1" << std::endl;

	//cudaSetDevice( 1 );

	gpuErrchk( cudaMallocHost( (void**) &original , filesize * sizeof( cufftReal ) ) );

	//std::cout << "2" << std::endl;

	for( unsigned int i = 0 ; i < filesize ; i++ )
	{
	
		original[i] = (cufftReal) (byte) f.get();

	}

	f.close();

	cudaGetSymbolAddress( (void**) &deviceCount , transmissionCount );

	gpuErrchk( cudaMalloc( &deviceSource , NUM_STREAMS * BLOCKS_PER_STREAM * BLOCK_SIZE * sizeof(cufftReal) ));

	gpuErrchk( cudaMalloc( &deviceResult ,  NUM_STREAMS * BLOCKS_PER_STREAM * BLOCK_SIZE * sizeof(cufftComplex) ));

	gpuErrchk( cudaMalloc( &deviceScaledResult ,  NUM_STREAMS * BLOCKS_PER_STREAM * BLOCK_SIZE * sizeof(cufftReal) ));

	gpuErrchk( cudaMalloc( &deviceBins , MAX_TRANSMISSIONS * sizeof(int) ));

	gpuErrchk( cudaMalloc( &deviceFrequencies ,  MAX_TRANSMISSIONS * sizeof(cufftReal) ));

	gpuErrchk( cudaMalloc( &deviceStarts , MAX_TRANSMISSIONS * sizeof(cufftReal) ));

	gpuErrchk( cudaMalloc( &deviceEnds ,  MAX_TRANSMISSIONS * sizeof(cufftReal) ));

	gpuErrchk( cudaMalloc( &deviceStrengths ,  MAX_TRANSMISSIONS * sizeof(cufftReal) ));

	gpuErrchk( cudaMalloc( &deviceActiveTransmissions ,  NUM_STREAMS * BLOCKS_PER_STREAM * BLOCK_SIZE * sizeof(bool) ));

	//std::cout << "3" << std::endl;

	initTransmissionArray<<< 64 , 32 >>>( deviceActiveTransmissions );

	//std::cout << "4" << std::endl;

	cudaStream_t streams[ NUM_STREAMS ];

	for( int i = 0 ; i < NUM_STREAMS ; i++ )
	{

		gpuErrchk( cudaStreamCreate( &streams[ i ] ) );

	}

	//std::cout << "5" << std::endl;

	//prepare the FFT
	cufftHandle plans[ NUM_STREAMS ];

	cufftResult_t fft_result;

	for( int i = 0 ; i < NUM_STREAMS ; i++ )
	{

		fft_result = cufftPlan1d( &plans[i] , BLOCK_SIZE , CUFFT_R2C , BATCH_SIZE );

		if( fft_result != CUFFT_SUCCESS )

			exit(1);

		fft_result = cufftSetStream( plans[i] , streams[i] );

		if( fft_result != CUFFT_SUCCESS )

			exit(1);

	}

	//std::cout << "6" << std::endl;

	for( unsigned int j = 0 ; j < filesize * 0.25 - BLOCK_SIZE * NUM_STREAMS  ; j += BLOCK_SIZE * NUM_STREAMS * BLOCKS_PER_STREAM )
	{
		int iteration_offset = j * sizeof( cufftReal );
		
		for( int k = 0 ; k < NUM_STREAMS ; k++ )
		{

			int stream_offset = k * BLOCK_SIZE * BLOCKS_PER_STREAM;

			cudaMemcpyAsync( deviceSource + stream_offset  , original + iteration_offset + stream_offset , BLOCK_SIZE * BLOCKS_PER_STREAM , cudaMemcpyHostToDevice , streams[ k ] );

			for( int l = 0 ; l < BLOCKS_PER_STREAM ; l++ )
			{

				int block_offset = l * BLOCK_SIZE;

				fft_result = cufftExecR2C( plans[k] , deviceSource + stream_offset + block_offset , deviceResult + stream_offset + block_offset );

				if( fft_result != CUFFT_SUCCESS )

					exit(2);

				// num blocks * num threads = fftsize / 2 ... nyquist limit
				scaleResult<<< 64 , 32 , 0 , streams[ k ] >>>( deviceScaledResult + stream_offset + block_offset , deviceResult + stream_offset + block_offset );

				gpuErrchk( cudaPeekAtLastError() );

				findTransmissions<<< 64 , 32 , 0 , streams[ k ] >>>(
						deviceScaledResult + stream_offset + block_offset,
						deviceBins,
						deviceFrequencies,
						deviceStarts,
						deviceEnds,
						deviceStrengths,
						deviceActiveTransmissions
					);

				//std::cout << "11" << std::endl;

				gpuErrchk( cudaPeekAtLastError() );

			}
		}

	}

	//std::cout << "12" << std::endl;

	//Copy all that crap back
	int* hostCount = new int;

	hostBins = new int[ MAX_TRANSMISSIONS ];

	hostFrequencies = new cufftReal[ MAX_TRANSMISSIONS ];

	hostStarts = new cufftReal[ MAX_TRANSMISSIONS ];

	hostEnds = new cufftReal[ MAX_TRANSMISSIONS ];

	hostStrengths = new cufftReal[ MAX_TRANSMISSIONS ];

	//std::cout << "13" << std::endl;

	gpuErrchk( cudaMemcpy( hostBins , deviceBins , MAX_TRANSMISSIONS * sizeof( int ) , cudaMemcpyDeviceToHost ));

	gpuErrchk( cudaMemcpy( hostFrequencies , deviceFrequencies , MAX_TRANSMISSIONS * sizeof( cufftReal ) , cudaMemcpyDeviceToHost ));

	gpuErrchk( cudaMemcpy( hostStarts , deviceStarts , MAX_TRANSMISSIONS * sizeof( cufftReal ) , cudaMemcpyDeviceToHost ));

	gpuErrchk( cudaMemcpy( hostEnds , deviceEnds , MAX_TRANSMISSIONS * sizeof( cufftReal ) , cudaMemcpyDeviceToHost ));

	gpuErrchk( cudaMemcpy( hostStrengths , deviceStrengths , MAX_TRANSMISSIONS * sizeof( cufftReal ) , cudaMemcpyDeviceToHost ));

	gpuErrchk( cudaMemcpy( hostCount , deviceCount , sizeof( int ) , cudaMemcpyDeviceToHost ) );

	//std::cout << "14" << std::endl;

	std::cout << *hostCount << std::endl;

	std::ofstream fo;

	fo.open( "spikes.txt" );

	for( unsigned int i = 0 ; i < *hostCount ; i++ )
	{

		fo << "==== TRANSMISSION ====" << "\n";

		//In MHz
		fo << "Bin             : " << hostBins[ i ] << " \n";
		fo << "Frequency       : " << hostFrequencies[ i ] << " MHz\n";
		fo << "Signal strength : " << hostStrengths[ i ] << " dB\n";
		fo << "Time start      : " << hostStarts[ i ] << " s\n";
		fo << "Time end        : " << hostEnds[ i ] << " s\n";

	}

	return 0;

}
