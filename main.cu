#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>
#include <map>
#include <stdexcept>

#define PI 3.14159265359
#define BLOCK_SIZE 4096
#define SIGNAL_THRESHOLD 102

//172MHz gives us CB
#define SAMPLE_RATE 172089331.259

#define HzInMHz 1000000

typedef char byte;

typedef double fftw_real;

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


	fftw_real* original = new fftw_real[ filesize ];

	fftw_complex* result = new fftw_complex[ filesize ];

	for( unsigned int i = 0 ; i < filesize ; i++ )
	{
	
		original[i] = (double) (byte) f.get();

	}

	f.close();

	int fft_size = BLOCK_SIZE;

	//create a new input for the windowing function
	fftw_real* currentWindow = new fftw_real[ fft_size ];
	
	fftw_real* resultScaled = new fftw_real[ fft_size ];

	for( unsigned int j = 0 ; j < filesize * 0.125 - BLOCK_SIZE  ; j += BLOCK_SIZE )
	{
		
		memcpy( currentWindow , original + j * sizeof(fftw_real) , BLOCK_SIZE * sizeof(fftw_real) );
		
		//prepare the FFT
		fftw_plan p = fftw_plan_dft_r2c_1d( fft_size , currentWindow , result , FFTW_ESTIMATE );

		//Run the FFT
		fftw_execute( p );

		//calculate amplitude of first N/2 bins (Nyquist Limit?)
		for( unsigned int i = 0 ; i < BLOCK_SIZE / 2 ; i++ )
		{

			//get magnitude using complex conjugate
			resultScaled[ i ] = sqrt( result[ i ][ 0 ] * result[ i ][ 0 ] + result[ i ][ 1 ] * result[ i ][ 1 ] );

			//put into decibels
			resultScaled[ i ] = 20 * log10( resultScaled[ i ] );

			//difference[ i ] = 20.0f * log10( resultScaled[ i ] - resultScaledOld[ i ] );

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
