#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "genann.h"

/* This example is to illustrate how to use GENANN.
 * It is NOT an example of good machine learning techniques.
 */

const char *iris_data = "example/iris.data";

double *input, *class;
int samples;
enum classes{
    IRIS_SETOSA,
    IRIS_VERSICOLOR,
    IRIS_VIRGINICA,
    classes_length
};
const char *class_names[classes_length] = {
    [IRIS_SETOSA] = "Iris-setosa", 
    [IRIS_VERSICOLOR] = "Iris-versicolor",
    [IRIS_VIRGINICA] = "Iris-virginica"};

void load_data() {
    /* Load the iris data-set. */
    FILE *in = fopen("example/iris.data", "r");
    if (!in) {
        printf("Could not open file: %s\n", iris_data);
        exit(1);
    }

    /* Loop through the data to get a count. */
    char line[1024];
    while (!feof(in) && fgets(line, 1024, in)) {
        ++samples;
    }
    fseek(in, 0, SEEK_SET);

    printf("Loading %d data points from %s\n", samples, iris_data);

    /* Allocate memory for input and output data. */
    input = malloc(sizeof(double) * samples * 4); // 4 inputs
    class = malloc(sizeof(double) * samples); // only 1 output 

    /* Read the file into our arrays. */
    int i, j;
    for (i = 0; i < samples; ++i) {
        double *p = input + i * 4;
        double *c = class + i;
        *c = 0.0;

        if (fgets(line, 1024, in) == NULL) {
            perror("fgets");
            exit(1);
        }

        char *split = strtok(line, ",");
        for (j = 0; j < 4; ++j) {
            p[j] = atof(split);
            split = strtok(0, ",");
        }

        split[strlen(split)-1] = 0;
        if (strcmp(split, class_names[0]) == 0) {c[0] = 1.0;}
        else if (strcmp(split, class_names[1]) == 0) {c[0] = 2.0;}
        else if (strcmp(split, class_names[2]) == 0) {c[0] = 3.0;}
        else {
            printf("Unknown class %s.\n", split);
            exit(1);
        }

        /* printf("Data point %d is %f %f %f %f  ->   %f %f %f\n", i, p[0], p[1], p[2], p[3], c[0], c[1], c[2]); */
    }

    fclose(in);
}


int main(int argc, char *argv[])
{
    printf("GENANN example 4.\n");
    printf("Train an ANN on the IRIS dataset using random search.\n");

    srand(time(0));

    /* Load the data from file. */
    load_data();

    /* 4 inputs.
     * 1 hidden layer(s) of 4 neurons.
     * 1 outputs (1 per class)
     */
    genann *ann = genann_init(4, 1, 4, 1);

    int i, j;
    int count = 0;
    double err,last_err = 0.0;
    do {
        ++count;
        if (count % 1000 == 0) {
            /* We're stuck, start over. */
            genann_randomize(ann);
            last_err = 1000;
        }

        genann *save = genann_copy(ann);

        /* Take a random guess at the ANN weights. */
        for (i = 0; i < ann->total_weights; ++i) {
            ann->weight[i] += ((double)rand())/RAND_MAX-0.5;
        }

        /* See how we did. */
        for (int j = 0; j < samples; ++j) {
            err += pow(*genann_run(ann, input + j*4) - class[j], 2.0);
        }

            /* Keep these weights if they're an improvement. */
            if (err < last_err) {
                genann_free(save);
                last_err = err;
            } else {
                genann_free(ann);
                ann = save;
            }

        } while (err > 0.01);

        int correct = 0;
        for (j = 0; j < samples; ++j) {
            const double *guess = genann_run(ann, input + j*4);
            correct += (*guess == class[j])? 1 : 0;
        }

        printf("%d/%d correct (%0.1f%%).\n", correct, samples, (double)correct / samples * 100.0);



        genann_free(ann);
        free(input);
        free(class);

        return 0;
    }
