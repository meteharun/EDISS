//Mete Harun Akcay, 12.01.2025
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512

// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// number of real galaxies
int    NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// number of simulated random galaxies
int    NoofSim;

unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int *d_histogram;


__device__ float arcToRad(float arcmin) {
    return (arcmin / 60.0f) * (3.141592654f / 180.0f);
}

__device__ float computeAngularDistance(float asc1, float asc2, float decl1, float decl2) {
  // Computes the angular distance between two points on a sphere given their right ascension and declination.
  
  float asc1_rad, asc2_rad, decl1_rad, decl2_rad, cosine, angle_rad;

  asc1_rad = arcToRad(asc1);
  asc2_rad = arcToRad(asc2);
  decl1_rad = arcToRad(decl1);
  decl2_rad = arcToRad(decl2);

  cosine = sinf(decl1_rad)*sinf(decl2_rad) + cosf(decl1_rad)*cosf(decl2_rad)*cosf(asc1_rad-asc2_rad);
  if (cosine > 1.0) cosine = 1.0;
  if (cosine < -1.0) cosine = -1.0;

  angle_rad = acosf(cosine);
  return angle_rad * 180 / M_PI;
}

__global__ void computeHistogram(float* right_asc1, float* right_asc2, float* decl1, float* decl2, int total_galaxies, unsigned int* histogram_bins) {
  // Calculates a histogram of angular distances between pairs of galaxies.

  long int global_thread_id = (long int)blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id >= (long int)total_galaxies * total_galaxies) return;

  int first_galaxy_id = global_thread_id / total_galaxies;
  int second_galaxy_id = global_thread_id % total_galaxies;

  float computed_angle = computeAngularDistance(
      right_asc1[first_galaxy_id], right_asc2[second_galaxy_id], decl1[first_galaxy_id], decl2[second_galaxy_id]
  );

  int histogram_bin_id = (int)(computed_angle / 0.25);
  atomicAdd(&histogram_bins[histogram_bin_id], 1);
}



int main(int argc, char *argv[])
{
  //int    i;
  int    noofblocks;
  int    readdata(char *argv1, char *argv2);
  int    getDevice(int deviceno);
  long int hist_DR_sum, hist_DD_sum, hist_RR_sum;
  double w;
  double start, end, kerneltime;
  struct timeval _ttime;
  struct timezone _tzone;
  //cudaError_t myError;

  FILE *outfil;

  if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}

  if ( getDevice(0) != 0 ) return(-1);

  if ( readdata(argv[1], argv[2]) != 0 ) return(-1);

  // allocate memory on the GPU
  size_t hist_size = totaldegrees*binsperdegree*sizeof(unsigned int);
  size_t buffer_size = NoofReal*sizeof(float);
  float *ra_1, *ra_2, *decl_1,  *decl_2;

  cudaMalloc(&ra_1, buffer_size);
  cudaMalloc(&ra_2, buffer_size);
  cudaMalloc(&decl_1, buffer_size);
  cudaMalloc(&decl_2, buffer_size);
  cudaMalloc(&d_histogram, hist_size);

  histogramDR = (unsigned int *)calloc(hist_size, sizeof(unsigned int));
  histogramDD = (unsigned int *)calloc(hist_size, sizeof(unsigned int));
  histogramRR = (unsigned int *)calloc(hist_size, sizeof(unsigned int));
  
  
  kerneltime = 0.0;
  gettimeofday(&_ttime, &_tzone);
  start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
  
  //DD
  // copy data to the GPU
  cudaMemcpy(ra_1, ra_real, buffer_size, cudaMemcpyHostToDevice);
  cudaMemcpy(ra_2, ra_real, buffer_size, cudaMemcpyHostToDevice);
  cudaMemcpy(decl_1, decl_real, buffer_size, cudaMemcpyHostToDevice);
  cudaMemcpy(decl_2, decl_real, buffer_size, cudaMemcpyHostToDevice);

  // initialize histogram
  cudaMemset(d_histogram, 0, hist_size);
  
  // calculate number of blocks
  noofblocks = ((long int)NoofReal*NoofReal + threadsperblock - 1) / threadsperblock;

  // run the kernels on the GPU
  computeHistogram<<<noofblocks, threadsperblock>>>(ra_1, ra_2, decl_1, decl_2, NoofReal, d_histogram);

  // copy the results back to the CPU
  cudaMemcpy(histogramDD, d_histogram, hist_size, cudaMemcpyDeviceToHost);

  //RR
  cudaMemcpy(ra_1, ra_sim, buffer_size, cudaMemcpyHostToDevice);
  cudaMemcpy(ra_2, ra_sim, buffer_size, cudaMemcpyHostToDevice);
  cudaMemcpy(decl_1, decl_sim, buffer_size, cudaMemcpyHostToDevice);
  cudaMemcpy(decl_2, decl_sim, buffer_size, cudaMemcpyHostToDevice);
  cudaMemset(d_histogram, 0, hist_size);
  computeHistogram<<<noofblocks, threadsperblock>>>(ra_1,ra_2, decl_1, decl_2, NoofReal, d_histogram);
  cudaMemcpy(histogramRR, d_histogram, hist_size, cudaMemcpyDeviceToHost);

  //DR
  cudaMemcpy(ra_1, ra_real, buffer_size, cudaMemcpyHostToDevice);
  cudaMemcpy(ra_2, ra_sim, buffer_size, cudaMemcpyHostToDevice);
  cudaMemcpy(decl_1, decl_real, buffer_size, cudaMemcpyHostToDevice);
  cudaMemcpy(decl_2, decl_sim, buffer_size, cudaMemcpyHostToDevice);
  cudaMemset(d_histogram, 0, hist_size);
  computeHistogram<<<noofblocks, threadsperblock>>>(ra_1, ra_2, decl_1, decl_2, NoofReal, d_histogram);
  cudaMemcpy(histogramDR, d_histogram, hist_size, cudaMemcpyDeviceToHost);

  //Free memory
  cudaFree(ra_1);
  cudaFree(ra_2);
  cudaFree(decl_1);
  cudaFree(decl_2);
  cudaFree(d_histogram);



  // calculate omega values on the CPU
  hist_DR_sum = 0;
  hist_DD_sum = 0;
  hist_RR_sum = 0;

  outfil = fopen(argv[3], "w");
  fprintf(outfil, "%8s %10s %10s %10s %10s\n", "bin", "hist_DR", "hist_DD", "hist_RR", "omega");
  for (int bin_idx = 0; bin_idx < hist_size; bin_idx++) {
    if (histogramRR[bin_idx] == 0) break;
    w = int(histogramDD[bin_idx] - 2*histogramDR[bin_idx] + histogramRR[bin_idx]) / float(histogramRR[bin_idx]);
    
    hist_DR_sum += histogramDR[bin_idx];    
    hist_DD_sum += histogramDD[bin_idx];
    hist_RR_sum += histogramRR[bin_idx];
    fprintf(outfil, "%8.2f %10d %10d %10d %10.5f\n", bin_idx * 0.25, histogramDR[bin_idx], histogramDD[bin_idx], histogramRR[bin_idx], w);
  }
  printf("DR histogram sum: %ld Ok!\n", hist_DR_sum);
  printf("DD histogram sum: %ld Ok!\n", hist_DD_sum);
  printf("RR histogram sum: %ld Ok!\n", hist_RR_sum);

  fclose(outfil);
  free(histogramDR);
  free(histogramDD);
  free(histogramRR);
  


  
  gettimeofday(&_ttime, &_tzone);
  end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
  kerneltime += end-start;
  printf("   Run time = %.lf secs\n",kerneltime);
  return(0);
}

int readdata(char *argv1, char *argv2)
{
  int i,linecount;
  char inbuf[180];
  double ra, dec;
  FILE *infil;
                                         
  printf("   Assuming input data is given in arc minutes!\n");
                          // spherical coordinates phi and theta in radians:
                          // phi   = ra/60.0 * dpi/180.0;
                          // theta = (90.0-dec/60.0)*dpi/180.0;

  //dpi = acos(-1.0);
  infil = fopen(argv1,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv1);return(-1);}

  // read the number of galaxies in the input file
  int announcednumber;
  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv1);return(-1);}
  linecount =0;
  while ( fgets(inbuf,180,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv1, linecount);
  else 
      {
      printf("   %s does not contain %d galaxies but %d\n",argv1, announcednumber,linecount);
      return(-1);
      }

  NoofReal = linecount;
  ra_real   = (float *)calloc(NoofReal,sizeof(float));
  decl_real = (float *)calloc(NoofReal,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i = 0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv1);
         fclose(infil);
         return(-1);
         }
      ra_real[i]   = (float)ra;
      decl_real[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofReal ) 
      {
      printf("   Cannot read %s correctly\n",argv1);
      return(-1);
      }

  infil = fopen(argv2,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv2);return(-1);}

  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv2);return(-1);}
  linecount =0;
  while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv2, linecount);
  else
      {
      printf("   %s does not contain %d galaxies but %d\n",argv2, announcednumber,linecount);
      return(-1);
      }

  NoofSim = linecount;
  ra_sim   = (float *)calloc(NoofSim,sizeof(float));
  decl_sim = (float *)calloc(NoofSim,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i =0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) 
         {
         printf("   Cannot read line %d in %s\n",i+1,argv2);
         fclose(infil);
         return(-1);
         }
      ra_sim[i]   = (float)ra;
      decl_sim[i] = (float)dec;
      ++i;
      }

  fclose(infil);

  if ( i != NoofSim ) 
      {
      printf("   Cannot read %s correctly\n",argv2);
      return(-1);
      }

  return(0);
}




int getDevice(int deviceNo)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
       printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                   =   %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels             =   ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(deviceNo);
    cudaGetDevice(&device);
    if ( device != deviceNo ) printf("   Unable to set device %d, using device %d instead",deviceNo, device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}