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
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define PI 3.14159265359
#define BLOCK_SIZE 4096
#define SIGNAL_THRESHOLD 102
#define MAX_TRANSMISSIONS 200

//172MHz gives us CB
#define SAMPLE_RATE 172089331.259
#define BATCH_SIZE 1

#define HzInMHz 1000000

typedef char byte;
typedef float2 Complex;

__device__ cufftReal* sourceBuffer;

__device__ cufftComplex* resultBuffer;

__device__ cufftReal* scaledResultBuffer;

__device__ bool* activeTransmissions;

__device__ int* transmissionBins;

__device__ cufftReal* transmissionFrequencies;

__device__ cufftReal* transmissionStarts;

__device__ cufftReal* transmissionEnds;

__device__ cufftReal* transmissionStrengths;

__device__ int transmissionCount;

__device__ int timeStep;

void __global__ scaleResult( )
{

	int idx = threadIdx.x;

	if( idx < BLOCK_SIZE )
	{

		scaledResultBuffer[ idx ] = sqrt( resultBuffer[ idx ][ 0 ] * resultBuffer[ idx ][ 0 ] * +
										  resultBuffer[ idx ][ 1 ] * resultBuffer[ idx ][ 1 ]      );

		scaledResultBuffer[ idx ] = 20 * log10( scaledResultBuffer[ idx ] );

	}

}

void __global__ findTransmissions( )
{

	int idx = threadIdx.x;

	if( idx < BLOCK_SIZE )
	{

		if( scaledResultBuffer[ idx ] > SIGNAL_THRESHOLD && activeTransmissions[ idx ] == false )
		{



		}

	}

}

void __device__ createTransmission( int idx )
{



}

cufftComplex *deviceResult;

cufftReal *deviceSource;

cufftReal *deviceScaledResult;

int* deviceBins;

cufftReal *deviceFrequencies;

cufftReal *deviceStarts;

cufftReal *deviceEnds;

cufftReal *deviceStrengths;

bool* deviceActiveTransmissions;

int* deviceCount;

void outputFFTData( std::string filename, fftw_real* data , unsigned int size );

class transmission
{

public:

	int bin;

	float frequency;

	float timeStart;

	float timeEnd;

	float peakStrength;

};

int main( int argc , char** argv )
{
	
	std::string filename = std::string( argv[1] );

	std::ifstream f;

	std::map< int , transmission > currentSpikes;

	std::vector< transmission > historicalSpikes;

	struct stat filestatus;

	stat( filename.c_str() , &filestatus );

	size_t filesize = filestatus.st_size;

	f.open( filename.c_str() , std::ios::in | std::ios::binary );

	if( !f.good() )
	{
	
		std::cerr << "Can't open file" << std::endl;

		exit( 1 );

	}

	cufftReal* original = new cufftReal[ filesize ];

	for( unsigned int i = 0 ; i < filesize ; i++ )
	{
	
		original[i] = (cufftReal) (byte) f.get();

	}

	f.close();

	int fft_size = BLOCK_SIZE;

	int max_transmissions = MAX_TRANSMISSIONS;

	//get the address for the device's source buffer
	cudaGetSymbolAddress( (void**) &deviceSource , sourceBuffer );

	//get the address for the device's result buffer
	cudaGetSymbolAddress( (void**) &deviceResult , resultBuffer );

	//get the address for the device's result buffer
	cudaGetSymbolAddress( (void**) &deviceScaledResult , scaledResultBuffer );

	//get the address for the device's result buffer
	cudaGetSymbolAddress( (void**) &deviceBins , transmissionBins );

	//get the address for the device's result buffer
	cudaGetSymbolAddress( (void**) &deviceFrequencies , transmissionFrequencies );

	//get the address for the device's result buffer
	cudaGetSymbolAddress( (void**) &deviceStarts , transmissionStarts );

	//get the address for the device's result buffer
	cudaGetSymbolAddress( (void**) &deviceEnds , transmissionEnds );

	//get the address for the device's result buffer
	cudaGetSymbolAddress( (void**) &deviceStrengths , transmissionStrengths );

	//get the address for the device's result buffer
	cudaGetSymbolAddress( (void**) &deviceCount , transmissionCount );

	//get the address for the device's result buffer
	cudaGetSymbolAddress( (void**) &deviceActiveTransmissions , activeTransmissions );

	cudaMalloc( &deviceSource , filesize * sizeof(cufftReal) );

	cudaMalloc( &deviceResult ,  fft_size * sizeof(cufftComplex) );

	cudaMalloc( &deviceScaledResult ,  fft_size * sizeof(cufftReal) );

	cudaMalloc( &deviceBins , max_transmissions * sizeof(int) );

	cudaMalloc( &deviceFrequencies ,  max_transmissions * sizeof(cufftReal) );

	cudaMalloc( &deviceStarts , max_transmissions * sizeof(cufftReal) );

	cudaMalloc( &deviceEnds ,  max_transmissions * sizeof(cufftReal) );

	cudaMalloc( &deviceStrengths ,  max_transmissions * sizeof(cufftReal) );

	cudaMalloc( &deviceActiveTransmissions ,  fft_size * sizeof(bool) );

	// TODO: This giant memcpy will become a pipelined streaming thingy
	cudaMemcpy( deviceSource , original , filesize * 0.25 * sizeof( cufftReal ) , cudaMemcpyHostToDevice );

	for( unsigned int j = 0 ; j < filesize * 0.25 - fft_size  ; j += fft_size )
	{
		
		//prepare the FFT
		cufftHandle p;

		cufftPlan1d( &p , BLOCK_SIZE , CUFFT_R2C , BATCH_SIZE );

		//Run the FFT
		cufftExecR2C( p , deviceSource, deviceResult );

		//calculate amplitude of first N/2 bins (Nyquist Limit?)
		for( unsigned int i = 0 ; i < fft_size / 2 ; i++ )
		{

			bool activeTransmission = true;

			try
			{

				currentSpikes.at( i );

			}
			catch( std::out_of_range& e )
			{

				activeTransmission = false;

			}

			if( resultScaled[ i ] > SIGNAL_THRESHOLD && activeTransmission == false )
			{
				
				transmission trans;

				trans.bin = i;

				//frequency in MHz
				trans.frequency = i * SAMPLE_RATE / fft_size / HzInMHz;

				trans.timeStart = j / SAMPLE_RATE;

				trans.peakStrength = resultScaled[ i ];

				currentSpikes.insert( std::pair< int, transmission >( i , trans ) );

				//debug
				outputFFTData( "spikeWindow.txt" , resultScaled , fft_size );

			}

			if( resultScaled[ i ] < SIGNAL_THRESHOLD && activeTransmission == true )
			{

				transmission t = currentSpikes.at( i );

				t.timeEnd = j / SAMPLE_RATE;

				historicalSpikes.push_back( t );

				currentSpikes.erase( i );

			}
		}

	}

	std::ofstream fo;

	fo.open( "spikes.txt" );

	for( unsigned int i = 0 ; i < historicalSpikes.size() ; i++ )
	{

		fo << "====TRANSMISSION====" << "\n";

		//In MHz
		fo << "Frequency       : " << historicalSpikes[ i ].frequency << " MHz\n";
		fo << "Signal strength : " << historicalSpikes[ i ].peakStrength << " dB\n";
		fo << "Time start      : " << historicalSpikes[ i ].timeStart << " s\n";
		fo << "Time end        : " << historicalSpikes[ i ].timeEnd << " s\n";

	}

	return 0;

}

void outputFFTData( std::string filename, fftw_real* data , unsigned int size )
{
	std::ofstream fo;

	fo.open( filename.c_str() );

	for( unsigned int i = 0 ; i < size * 0.5 ; i++ )
	{

		fo << data[ i ] << std::endl;

	}

}
