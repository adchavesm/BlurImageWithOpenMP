//Librerias estándar de C
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>
#include <string.h>

//Libreria de Open MP
#include "omp.h"

//Librerias de terceros usadas para manipulación de imágenes png, jpg, etc.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_library/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_library/stb_image_write.h"
  

//Función para generar el kernel gaussiano
void generate_kernel(int size, double** kernel) 
{   
    //Desviación estándar 
    double sigma = 15.0; 
    
    //Suma para normalizar el kernel después 
    double sum = 0.0; 
    
    int mid = size/2;

    //Generar kernel de tamaño size x size usando funcion de densidad de probabilidad de Gauss
    for (int x = -mid; x <= mid; x++) { 
        for (int y = -mid; y <= mid; y++) { 
            kernel[x + mid][y + mid] = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma); 
            sum += kernel[x + mid][y + mid]; 
        } 
    } 
  
    //Normalización del kernel para un efecto mas limpio en el blurring
    for (int i = 0; i < size; ++i) 
        for (int j = 0; j < size; ++j) 
            kernel[i][j] /= sum; 
} 


int main(int argc, char* argv[]) {

    int width, height, channels;

    //Verificación de cantidad de argumentos correcta
    if(argc != 5) {
        perror("Cantidad de argumentos no es valida!");
        return EXIT_FAILURE;
    }

    //Cargamos la imagen obteniendo sus datos
    unsigned char* img = stbi_load(argv[1], &width, &height, &channels, 0);

    //Verificación de imágen válida
    if(img == NULL) {
        perror("Error cargando la imagen!\n");
        return EXIT_FAILURE;
    }

    printf("\nancho: %dpx, alto: %dpx, canales: %d\n", width, height, channels);

    int kernel_size;

    //Extracción de tamaño del kernel 
    kernel_size = atoi(argv[3]);

    //Tamaño de kernel debe ser impar
    if(kernel_size % 2 == 0){

        perror("Tamaño de kernel debe ser impar!\n");
        return EXIT_FAILURE;
    }

    int mid_size = kernel_size / 2 ;

    //Asignación dinámica de espacio para generar una matriz 
    double** kernel = (double**)malloc(sizeof(double*) * kernel_size);
    for(size_t i = 0; i < kernel_size; ++i) 
        kernel[i] = (double*)malloc(sizeof(double)*kernel_size);
    
    generate_kernel(kernel_size, kernel);

    printf("\nKernel gaussiano usado para el filtro: \n\n");
    for(size_t i = 0; i < kernel_size; ++i) {

        for(size_t j = 0; j < kernel_size; ++j) 
            printf("%f ", kernel[i][j]);
        
        printf("\n");
    }

    size_t n_threads = atoi(argv[4]);
    

    //Construcción de tamaño de imágen que se quiere producir
    size_t img_size = width * height * channels;
    size_t blur_channels = 3;
    size_t blurred_image_size = width * height * blur_channels;

    //Calculo de bloques de pixeles a procesar
    const size_t chunk = (width * height)/(4*n_threads); 

    printf("\nTamaño de chunk de pixeles es: %li",chunk);
    unsigned char* blurred_img = (unsigned char*)malloc(sizeof(unsigned char) * blurred_image_size);
   
    struct timeval start, end;
   
    //Calculo de tiempo antes de iniciar operaciones de convolución
	gettimeofday(&start, NULL);

    //Establecemos número de threads a usar
    omp_set_num_threads(n_threads);

    //Loop que itera sobre un rango de pixeles para ejecutar las convoluciones
    //Directiva de openmp para repartir la carga de forma automática, usando schedule dinámico
    #pragma omp parallel for schedule(dynamic, chunk)
    for (int current_pixel = 0; current_pixel < width*height; ++current_pixel) {
        
        double valueRed = 0;
        double valueGreen = 0;
        double valueBlue = 0;

            //Recorrido por cada uno de los valores de la matriz del kernel
            for(int i = -mid_size; i <= mid_size; ++i){
                for(int j = -mid_size; j <= mid_size; ++j){
                    //Calculo de pixel de imágen original sobre el que queremos aplicar convolución
                    int target_pixel = current_pixel + (i*width) + j;

                    //Extracción de cada uno de los tres canales del pixel identificado
                    int pixel_valueRed = target_pixel < 0 || target_pixel > width*height-1 ? 1 : *(img+(target_pixel*channels)+0);
                    int pixel_valueGreen = target_pixel < 0 || target_pixel > width*height-1 ? 1 : *(img+(target_pixel*channels)+1);
                    int pixel_valueBlue = target_pixel < 0 || target_pixel > width*height-1 ? 1 : *(img+(target_pixel*channels)+2);
                    //printf("%i\n",target_pixel);

                    //Suma de valores multiplicados
                    valueRed += kernel[i+mid_size][j+mid_size] * pixel_valueRed;
                    valueGreen += kernel[i+mid_size][j+mid_size] * pixel_valueGreen;
                    valueBlue += kernel[i+mid_size][j+mid_size] * pixel_valueBlue;
                    //printf("%f\n", kernel[i+mid_size][j+mid_size]);
                }
            } 

        //Asignación de canales modificados a la imágen con filtro
        *(blurred_img+(current_pixel*channels) + 0)=  (uint8_t)(valueRed);
        *(blurred_img+(current_pixel*channels) + 1)=  (uint8_t)(valueGreen);
        *(blurred_img+(current_pixel*channels) + 2)=  (uint8_t)(valueBlue);  

        //printf("%i\n", *(blurred_img+(current_pixel*channels) + 0));

    }

    //Calculo de tiempo después de aplicar convolución a todos los pixeles de la imágen
    gettimeofday(&end, NULL);
    
    //Escribimos la imágen en formato jpg con el filtro aplicado
    stbi_write_jpg(argv[2], width, height, 3, blurred_img, 100);

    //Liberación de espacio usado por el kernel
    for(size_t i = 0; i < kernel_size; ++i) 
        free(kernel[i]);
    
    free(kernel);

    //Liberación de espacio usado para codificación de la imágen
    stbi_image_free(img);

    //Liberación de espacio usado para codificación de imágen con filtro
    free(blurred_img);

    //Tiempo total(Elapsed Wall time)
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    double seconds_d = (double)micros / pow(10,6);  

    //Devolver tiempo de ejecución del programa
    printf("\nTiempo de ejecucion: %f segundos\n", seconds_d);

    FILE * fp;
    int i;  
    size_t file_length = strlen(argv[1]) + strlen(argv[3]) + 6;
    char * fileName = (char *)malloc(sizeof(char)*file_length);
    fileName[0] = '\0';
    strcat(fileName, argv[1]);
    strcat(fileName, "_");
    strcat(fileName, argv[3]);
    strcat(fileName, ".txt");

    //Abrir archivo para registrar el tiempo medido
    fp = fopen (fileName,"a");
    
    fprintf (fp, "%f ", seconds_d);
   
    fclose (fp);
   
    return EXIT_SUCCESS;
}