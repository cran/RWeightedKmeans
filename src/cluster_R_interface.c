#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "cluster.h"

// R inlcudes
#include <R.h> 
#include <Rinternals.h>

SEXP C_kmeans(SEXP dat, SEXP cen, SEXP n, SEXP p, SEXP k)
{
    SEXP clustering = PROTECT(allocVector(INTSXP, asInteger(n))); 

    data d[asInteger(n)]; // Data
    uint32_t i,j;

    // Allocate memory for data management
    for(i=0;i<asInteger(n);i++)
    {
        d[i].ind = i;
        d[i].clusterID = -1;
        d[i].pred = NULL;
        d[i].succ = NULL;
        d[i].dim = malloc(asInteger(p)*sizeof(double));
        if(d[i].dim == NULL)
        {
            // Nothing to do
        }
    }

    for(j=0;j<asInteger(p);j++)
    {
        for(i=0;i<asInteger(n);i++)
        {
            d[i].dim[j] = REAL(dat)[asInteger(n)*j + (i)];
        }
    }

    cluster c[asInteger(k)];

    // Allocate clusters dimension memory
    CLUSTER_initClusters(asInteger(p), c, asInteger(k), INTEGER(cen), d);

    // Compute kmeans algorithm
    CLUSTER_kmeans(d, asInteger(n), asInteger(p), asInteger(k), c);

    // Free clusters dimension memory
    CLUSTER_freeClusters(c, asInteger(k));

    // Retrieve algortihm outputs
    for(i=0;i<asInteger(n);i++)
    {
        INTEGER(clustering)[i] = d[i].clusterID; // Retrieve clustering result
        free(d[i].dim); // Free allocated memory
    }

    UNPROTECT(1);
    return clustering;
}

SEXP C_weightedKmeans(SEXP dat, SEXP cen, SEXP n, SEXP p, SEXP k, SEXP ifc, SEXP ioc, SEXP fwm, SEXP owm)
{
    SEXP clustering = PROTECT(allocVector(INTSXP, asInteger(n))); 
    SEXP ow = PROTECT(allocVector(REALSXP, asInteger(n)));

    data d[asInteger(n)]; // Data
    uint32_t i,j;

    eMethodType objWeiMet = METHOD_SILHOUETTE; // Objects weights claculation method 
    eMethodType feaWeiMet = METHOD_DISPERSION; // Features weights claculation method

    // Allocate memory for data management
    for(i=0;i<asInteger(n);i++)
    {
        d[i].ind = i;
        d[i].clusterID = -1;
        d[i].pred = NULL;
        d[i].succ = NULL;
        d[i].dim = malloc(asInteger(p)*sizeof(double));
        if(d[i].dim == NULL)
        {
            // Nothing to do
        }
    }

    // Specific order for milligan function data
    for(j=0;j<asInteger(p);j++)
    {
        for(i=0;i<asInteger(n);i++)
        {
            d[i].dim[j] = REAL(dat)[asInteger(n)*j + (i)];
        }
    }

    // Retreive objects weights calculation method
    if(!strcmp("SIL", CHAR(asChar(owm))))
    {
        objWeiMet = METHOD_SILHOUETTE;
    }
    else if(!strcmp("SIL_NK", CHAR(asChar(owm))))
    {
        objWeiMet = METHOD_SILHOUETTE_NK;
    }
    else if(!strcmp("MED", CHAR(asChar(owm))))
    {
        objWeiMet = METHOD_MEDIAN;
    }
    else if(!strcmp("MED_NK", CHAR(asChar(owm))))
    {
        objWeiMet = METHOD_MEDIAN_NK;
    }
    else if(!strcmp("MIN_CEN_DIST", CHAR(asChar(owm))))
    {
        objWeiMet = METHOD_MIN_DIST_CENTROID;
    }
    else if(!strcmp("MIN_CEN_DIST_NK", CHAR(asChar(owm))))
    {
        objWeiMet = METHOD_MIN_DIST_CENTROID_NK;
    }
    else if(!strcmp("SUM_DIST_CEN", CHAR(asChar(owm))))
    {
        objWeiMet = METHOD_SUM_DIST_CENTROID;
    }
    else
    {
        // Unknown method. Use default.
        objWeiMet = METHOD_SILHOUETTE;
    }

    // Retreive features weights calculation method
    if(!strcmp("DISP", CHAR(asChar(fwm))))
    {
        feaWeiMet = METHOD_DISPERSION;
    }
    else
    {
        // Unknown method. Use default.
        feaWeiMet = METHOD_DISPERSION;
    }

    // Calculate the matrix of distance between points
    double **dist;
    CLUSTER_ComputeMatDistPointToPoint(d, asInteger(n), asInteger(p), &dist);

    cluster c[asInteger(k)];

    // Allocate clusters dimension memory
    CLUSTER_initClusters(asInteger(p), c, asInteger(k), INTEGER(cen), d);

    // Initialize object weights
    CLUSTER_initObjectWeights(d, asInteger(n));

    // Weighted feature k-means
    if(asLogical(ifc) && !asLogical(ioc))
    {
        CLUSTER_featuresWeightedKmeans(d, asInteger(n), asInteger(p), asInteger(k), c, asLogical(ifc), feaWeiMet);
    }
    // Weighted object or weighted object and weighted feature k-means
    else
    {
        // Compute kmeans algorithm
        CLUSTER_weightedKmeans(d, asInteger(n), asInteger(p), asInteger(k), c, asLogical(ifc), feaWeiMet, asLogical(ioc), objWeiMet, dist);
    }

    // Free clusters dimension memory
    CLUSTER_freeClusters(c, asInteger(k));

    // Free allocated distances matrix
    CLUSTER_FreeMatDistPointToPoint(asInteger(n), &dist); 

    // Retrieve algortihm outputs
    for(i=0;i<asInteger(n);i++)
    {
        INTEGER(clustering)[i] = d[i].clusterID; // Retrieve clustering result
        REAL(ow)[i] = d[i].ow; // Retrieve resulting object weights
        free(d[i].dim); // Free allocated memory
    }

    // Compile results
    SEXP vec = PROTECT(allocVector(VECSXP, 2));
    SET_VECTOR_ELT(vec, 0, clustering);
    SET_VECTOR_ELT(vec, 1, ow);

    UNPROTECT(3);
    return vec;
}
