/** @file cluster.c
 *  @brief Function declarations for the k-means clustering.
 *
 *  This contains the function declarations for the k-means
 *  clustering and eventually any macros, constants,or 
 *  global variables.
 *
 *  @author Alexandre Gondeau
 *  @bug No known bugs.
 */

/* -- Includes -- */

/* libc includes. */
#include <stdlib.h>

/* math header file. */
#include <math.h>

/* cluster header file. */
#include "cluster.h"

/* -- Defines -- */

/* Clustering constant defines. */
#define NB_ITER 100

/* Macro defines. */
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

/* -- Enumerations -- */

/** @brief Contains the different distance calculation types.
 *
 */
typedef enum _eDistanceType 
{
    DISTANCE_EUCLIDEAN = 0,
    DISTANCE_OTHER
} eDistanceType;

/** @brief Proceeds to a fake assigenment of data to  
 *         the different clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_fakeDataAssignmentToCentroids(data *dat, uint64_t n, cluster *c, uint32_t k);

/** @brief Chooses random data as centroids.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_randomCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Assigns data to the nearest centroid.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @param conv The pointer to a boolean specifying if the algorithm
 *              has converged. 
 *  @return Void.
 */
static double CLUSTER_assignDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv);

/** @brief Assigns and transfer data to the nearest cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @param conv The pointer to a boolean specifying if the algorithm
 *              has converged.
 *  @return Void.
 */
static void CLUSTER_transferDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv);

/** @brief Assigns weighted data to the nearest centroid
 *         for features weighted algorithm.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param conv The pointer to a boolean specifying if the algorithm
 *              has converged.
 *  @return Void.
 */
static void CLUSTER_assignFeaturesWeightedDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv);

/** @brief Assigns weighted data to the nearest centroid
 *         for objects (and features) weighted algorithm.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param internalFeatureWeights The boolean specifying if 
 *              the features weights are computed internally.
 *  @param  feaWeiMet The method used to computed the features
 *                    weights internally.
 *  @param internalObjectsWeights The boolean specifying if 
 *              the objects weights are computed internally.
 *  @param  objWeiMet The method used to computed the objects
 *                    weights internally.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @param wss The array of wss per cluster.
 *  @param conv The pointer to a boolean specifying if the algorithm
 *              has converged.
 *  @return Void.
 */
static void CLUSTER_assignWeightedDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, eMethodType feaWeiMet, bool internalObjectWeights, eMethodType objWeiMet, double **dist, double wss[k], bool *conv);

/** @brief Computes the squared distance between a point and a cluster.
 *
 *  @param dat The pointer to the datum.
 *  @param p The number of datum dimensions.
 *  @param c The pointer to the cluster.
 *  @param d The type of distance calculation. 
 *  @return the computed distance between the point and the cluster.
 */
static double CLUSTER_computeSquaredDistancePointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d);

/** @brief Computes the squared and features weighted
 *         distance between a point and a cluster.
 *
 *  @param dat The pointer to the datum.
 *  @param p The number of datum dimensions.
 *  @param c The pointer to the cluster.
 *  @param d The type of distance calculation. 
 *  @return the computed distance between the point and the cluster.
 */
static double CLUSTER_computeSquaredFWDistancePointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d);

/** @brief Computes the squared distance between two clusters.
 *
 *  @param dat The pointer to the datum.
 *  @param p The number of datum dimensions.
 *  @param ci The pointer to the cluster i.
 *  @param cj The pointer to the cluster j.
 *  @param d The type of distance calculation. 
 *  @return the computed distance between the point and the cluster.
 */
static double CLUSTER_computeSquaredDistanceClusterToCluster(cluster *ci, cluster *cj, uint64_t p, eDistanceType d);

/** @brief Computes the squared distance between a weighted 
 *         point and a cluster.
 *
 *  @param dat The pointer to the datum.
 *  @param p The number of datum dimensions.
 *  @param c The pointer to the cluster.
 *  @param d The type of distance calculation. 
 *  @param fw The pointer to the features weights.
 *  @param ow The point weight.
 *  @return The computed distance between the weigted point 
 *          and the cluster.
 */
static double CLUSTER_computeSquaredDistanceWeightedPointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d, double *fw, double ow);

/** @brief Updates the centroids positions.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return Void.
 */
static void CLUSTER_computeCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Transfer a point from a cluster to another.
 *
 *  @param dat The pointer to data.
 *  @param n The id of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the cluster.
 *  @param k The id of the cluster. 
 *  @return Void.
 */
static void CLUSTER_transferPointToCluster(data *dat, uint64_t indN, uint64_t p, cluster *c, uint32_t indK);

/** @brief Add a point to a cluster.
 *
 *  @param dat The pointer to data.
 *  @param c The pointer to the cluster.
 *  @return Void.
 */
static void CLUSTER_addPointToCluster(data *dat, cluster *c);

/** @brief Remove a point from a cluster.
 *
 *  @param dat The pointer to data.
 *  @param c The pointer to the cluster.
 *  @return Void.
 */
static void CLUSTER_removePointFromCluster(data *dat, cluster *c);

/** @brief Computes the silhouette score for a 
 *         clustering.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return The computed silhouette score for the clustering.
 */
static double CLUSTER_computeSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist);

/** @brief Computes the distance between a point
 *         and an other point .
 *
 *  @param iDat The pointer to the first point.
 *  @param jDat The pointer to the second point.
 *  @param p The number of data dimensions.
 *  @param d The type of distance calculation. 
 *  @return The computed distance between the first point 
 *          and the second point.
 */
static double CLUSTER_computeDistancePointToPoint(data *iDat, data *jDat, uint64_t p, eDistanceType d);

/** @brief Computes the variance ratio criterion for 
 *         a clustering.
 *
 *  @param dat The pointer to data.
 *  @param c The pointer to the clusters.
 *  @param SSE The sum of squared errors for the clustering.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @return The computed variance ratio criterion for the 
 *          clustering.
 */
static double CLUSTER_computeCH(data *dat, cluster *c, double SSE, uint64_t n, uint64_t p, uint32_t k);

/** @brief Computes features weights via different
 *         methods.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param m The method for features weights calculation.
 *  @return Void.
 */
static void CLUSTER_computeFeatureWeights(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, eMethodType m);

/** @brief Computes features weights in a specific 
 *         cluster via different methods.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The id of the cluster.
 *  @param fw The features weights.
 *  @param m The method for objects weights calculation.
 *  @return Void.
 */
static void CLUSTER_computeFeatureWeightsInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, eMethodType m);

/** @brief Computes features weights via dispersion
 *         score.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param fw The features weights.
 *  @param norm The norm from Lp-spaces.
 *  @return Void.
 */
static void CLUSTER_computeFeatureWeightsViaDispersion(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint8_t norm);

/** @brief Computes features weights in a specific 
 *         cluster via dispersion score.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param fw The features weights.
 *  @param norm The norm from Lp-spaces.
 *  @return Void.
 */
static void CLUSTER_computeFeatureWeightsInClusterViaDispersion(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, uint8_t norm);

/** @brief Computes feature dispersion.
 *
 *  @param dat The pointer to a datum.
 *  @param p The specific dimension.
 *  @param c The pointer to the cluster.
 *  @param norm The norm from Lp-spaces.
 *  @return The computed dispersion.
 */
static double CLUSTER_computeFeatureDispersion(data *dat, uint64_t p, cluster *c, uint8_t norm);

/** @brief Computes objects weights via different
 *         methods.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param m The method for objects weights calculation.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeights(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, eMethodType m, double **dist);

/** @brief Computes objects weights via different
 *         methods.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The id of the cluster.
 *  @param m The method for objects weights calculation.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, eMethodType m, double **dist);

/** @brief Computes objects weights via silhouette
 *         score.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist);

/** @brief Computes objects weights via silhouette
 *         score. The sum of objects weights in a cluster
 *         is equal to the number of points in the cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaSilhouetteNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist);

/** @brief Computes objects weights via silhouette
 *         score in a cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The index of the cluster.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double **dist);

/** @brief Computes objects weights via silhouette
 *         score in a cluster. The sum of objects weights in 
 *         the cluster is equal to the number of points in the 
 *         cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The index of the cluster.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaSilhouetteNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double **dist);

/** @brief Computes objects weights via the distance 
 *         with the nearest centroid (different from its own).
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaMinDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes objects weights via the distance 
 *         with the nearest centroid (different from its own). 
 *         The sum of objects weights in a cluster is equal to 
 *         the number of points in the cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaMinDistCentroidNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes objects weights via the distance 
 *         with the nearest centroid (different from its own).
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The index of the cluster.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaMinDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK);

/** @brief Computes objects weights via the distance 
 *         with the nearest centroid (different from its own) 
 *         in a cluster. The sum of objects weights in the 
 *         cluster is equal to the number of points in the cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The index of the cluster.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaMinDistCentroidNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK);

/** @brief Computes objects weights via the sum of distances 
 *         with the other centroids (different from its own).
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaSumDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes objects weights via the sum of distances 
 *         with the other centroids (different from its own)
 *         in a cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaSumDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK);

/** @brief Computes objects weights via the median.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaMedian(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes objects weights via the median. 
 *         The sum of objects weights in the cluster 
 *         is equal to the number of objects in the cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsViaMedianNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes objects weights via the median in a cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The index of the cluster.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaMedian(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t indK);

/** @brief Computes objects weights via the median in a cluster.
 *         The sum of objects weights in the cluster 
 *         is equal to the number of objects in the cluster.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters.
 *  @param indK The index of the cluster.
 *  @return Void.
 */
static void CLUSTER_computeObjectWeightsInClusterViaMedianNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t indK);

/** @brief Computes the sum of squared errors for a 
 *         clustering. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @return The computed sum of squared errors for a 
 *          clustering.
 */
static double CLUSTER_computeSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k);

/** @brief Computes the within sum of squares. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param wss The array of wss per cluster.
 *  @return Void.
 */
static void CLUSTER_computeNkWeightedWSS(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double wss[k]);

/** @brief Computes the within sum of squares. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param indK The cluster index.
 *  @return The computed wss for the cluster indK.
 */
static double CLUSTER_computeNkWeightedWSSInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK);

/* -- Function definitions -- */
double CLUSTER_kmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c)
{
    if(dat == NULL || n < 2 || p < 1 || k < 2)
    {
        return -1.0;
    }
    else
    {
        bool conv = false; // Has converged

        // Initialization
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, c, k);
        CLUSTER_assignDataToCentroids(dat, n, p, c, k, &conv);
        CLUSTER_computeCentroids(dat, n, p, c, k);

        uint8_t iter = 0;
        conv = false;
        while(iter < NB_ITER && conv == false)
        {
            // Data assignation
            CLUSTER_transferDataToCentroids(dat, n, p, c, k, &conv);
            iter++;
        }

        // Return the sum of squared errors
        return CLUSTER_computeSSE(dat, n, p, c, k); 
    }
}

double CLUSTER_featuresWeightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, eMethodType featureWeightsMethod)
{
    if(dat == NULL || n < 2 || p < 1 || k < 2)
    {
        return -1.0;
    }
    else
    {
        bool conv = false; // Has converged

        // Initialization
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, c, k);
        CLUSTER_assignDataToCentroids(dat, n, p, c, k, &conv);
        CLUSTER_computeCentroids(dat, n, p, c, k);

        // Computes weights
        if(internalFeatureWeights == true)
        {
            CLUSTER_computeFeatureWeights(dat, n, p, c, k, featureWeightsMethod);
        }

        uint8_t iter = 0;
        conv = false;
        while(iter < NB_ITER && conv == false)
        {
            // Data assignation
            CLUSTER_assignFeaturesWeightedDataToCentroids(dat, n, p, c, k, &conv); 
            
            // Update centroids
            CLUSTER_computeCentroids(dat, n, p, c, k);

            // Update feature weights
            if(internalFeatureWeights == true)
            {
                CLUSTER_computeFeatureWeights(dat, n, p, c, k, featureWeightsMethod);
            }

            iter++;
        }

        // Return the sum of squared errors
        return CLUSTER_computeSSE(dat, n, p, c, k); 
    }
}

double CLUSTER_weightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, eMethodType featureWeightsMethod, bool internalObjectWeights, eMethodType objectWeightsMethod, double **dist)
{
    if(!(dat == NULL || n < 2 || p < 1 || k < 2))
    {
        double wss[k];
        bool conv = false; // Has converged

        // Initialization
        CLUSTER_fakeDataAssignmentToCentroids(dat, n, c, k);
        CLUSTER_assignDataToCentroids(dat, n, p, c, k, &conv);
        CLUSTER_computeCentroids(dat, n, p, c, k);

        // Compute weights
        if(internalFeatureWeights == true)
        {
            CLUSTER_computeFeatureWeights(dat, n, p, c, k, featureWeightsMethod);
        }
        if(internalObjectWeights == true)
        {
            CLUSTER_computeObjectWeights(dat, n, p, c, k, objectWeightsMethod, dist);
        }

        CLUSTER_computeNkWeightedWSS(dat, n, p, c, k, wss);

        uint8_t iter = 0;
        conv = false;
        while(iter < NB_ITER && conv == false)
        {
            // Data assignation
            CLUSTER_assignWeightedDataToCentroids(dat, n, p, c, k, internalFeatureWeights, featureWeightsMethod, internalObjectWeights, objectWeightsMethod, dist, wss, &conv);

            iter++;
        }

        // Update weights
        if(internalFeatureWeights == true)
        {
            CLUSTER_computeFeatureWeights(dat, n, p, c, k, featureWeightsMethod);
        }
        if(internalObjectWeights == true)
        {
            CLUSTER_computeObjectWeights(dat, n, p, c, k, objectWeightsMethod, dist);
        }

        // Return the sum of squared errors
        return CLUSTER_computeSSE(dat, n, p, c, k); 
    }
    else
    
    {
        return -1.0;
    }
}

static void CLUSTER_fakeDataAssignmentToCentroids(data *dat, uint64_t n, cluster *c, uint32_t k)
{
    uint64_t i;
    uint32_t l = 0;
    for(i=0;i<n;i++)
    {
        // Assign fake cluster to avoid null cluster
        if(l == k)
            l = 0;

        CLUSTER_addPointToCluster(&(dat[i]), &(c[l]));
        l++;
    }
}

void CLUSTER_initClusters(uint64_t p, cluster *c, uint32_t k, int32_t *cen, data *dat)
{
    if(p < 1 || c == NULL || k < 2)
    {
        // Nothing to do
    }
    else
    {
        uint32_t l;
        uint64_t j;
        for(l=0;l<k;l++)
        {
            c[l].ind = l; // Init index
            c[l].centroid = malloc(p*sizeof(double)); // Allocate cluster dimension memory
            if(c[l].centroid == NULL)
            {
                // Nothing to do
            }
            else
            {
                c[l].nbData = 0;
                c[l].head = NULL;

                c[l].fw = malloc(p*sizeof(double)); // Allocate cluster features weights memory
                if(c[l].fw == NULL)
                {
                    // Nothing to do
                }
                else
                {
                    // Init centroid feature weights and dimensions
                    for(j=0;j<p;j++)
                    {
                        c[l].fw[j] = 1.0;
                        c[l].centroid[j] = dat[cen[l]].dim[j];
                    }
                }
            }
        }
    }
}

void CLUSTER_freeClusters(cluster *c, uint32_t k)
{
    if(c == NULL || k < 2)
    {
        // Nothing to do
    }
    else
    {
        uint32_t l;
        for(l=0;l<k;l++)
        {
            free(c[l].centroid); // Free cluster dimension memory
            free(c[l].fw); // Free cluster features weights memory
        }
    }
}

static void CLUSTER_randomCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint32_t l;
    uint64_t j;

    uint64_t rnd[k];
    for(l=0;l<k;l++)
    {
        // Avoid duplicate
        uint64_t randInd;
        bool ok = false;
        while(ok == false)
        {
            // Use random data as centroids
            randInd = rand() % n;
            ok = true;
            rnd[l] = randInd;
            uint8_t i;
            for(i=0;i<l;i++)
            {
                if(randInd == rnd[i])
                {
                    ok = false;
                }
            }
        }
        for(j=0;j<p;j++)
            c[l].centroid[j] = dat[randInd].dim[j];
    }
}

static double CLUSTER_assignDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv)
{
    uint64_t i;
    uint32_t l;
    double SSE = 0.0;

    // Set convergence variable
    *conv = true;

    for(i=0;i<n;i++)
    {
        double minDist;
        uint32_t minK;
        for(l=0;l<k;l++)
        {
            // Calculate squared Euclidean distance
            double dist = CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN);

            if(l == 0)
            {
                minDist = dist;
                minK = l;
            }
            else
            {
                if(dist < minDist)
                {
                    minDist = dist;
                    minK = l; // Save the cluster for the min distance
                }
            }
        }

        if(minK != dat[i].clusterID)
        {
            // Remove point from former cluster
            CLUSTER_removePointFromCluster(&(dat[i]), &(c[dat[i].clusterID]));
            // Add point to cluster minK
            CLUSTER_addPointToCluster(&(dat[i]), &(c[minK]));

            // Reset convergence variable
            *conv = false;
        }

        SSE += minDist;
    }

    return SSE;
}

static void CLUSTER_transferDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2 || conv == NULL)
    {
        // Nothing to do
    }
    else
    {
        // MacQueen algorithm //
        
        uint64_t i;
        uint32_t l;

        // Set convergence variable
        *conv = true;

        for(i=0;i<n;i++)
        {
            double minDist;
            uint32_t minK;

            // For each cluster
            for(l=0;l<k;l++)
            {
                // Calculate squared Euclidean distance
                double dist = CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN);

                if(l == 0)
                {
                    minDist = dist;
                    minK = l;
                }
                else
                {
                    if(dist < minDist)
                    {
                        minDist = dist;
                        minK = l; // Save the cluster for the min distance
                    }
                }
            }

            if(minK != dat[i].clusterID)
            {
                // Transfer point i to cluster minK
                CLUSTER_transferPointToCluster(dat, i, p, c, minK);

                // Reset convergence variable
                *conv = false;
            }
        }
    }
}

static void CLUSTER_assignFeaturesWeightedDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool *conv)
{
    uint64_t i;
    uint32_t l;

    // Set convergence variable
    *conv = true;

    for(i=0;i<n;i++)
    {
        double minDist;
        uint32_t minK;
        for(l=0;l<k;l++)
        {
            // Calculate features weighted squared Euclidean distance
            double dist = CLUSTER_computeSquaredFWDistancePointToCluster(&(dat[i]), p, &(c[l]), DISTANCE_EUCLIDEAN);

            if(l == 0)
            {
                minDist = dist;
                minK = l;
            }
            else
            {
                if(dist < minDist)
                {
                    minDist = dist;
                    minK = l; // Save the cluster for the min distance
                }
            }
        }

        if(minK != dat[i].clusterID)
        {
            // Remove point from former cluster
            CLUSTER_removePointFromCluster(&(dat[i]), &(c[dat[i].clusterID]));
            // Add point to cluster minK
            CLUSTER_addPointToCluster(&(dat[i]), &(c[minK]));

            // Reset convergence variable
            *conv = false;
        }
    }
}

static void CLUSTER_assignWeightedDataToCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, bool internalFeatureWeights, eMethodType feaWeiMet, bool internalObjectWeights, eMethodType objWeiMet, double **dist, double wss[k], bool *conv)
{
    uint64_t i;
    uint32_t l;

    // Set convergence variable
    *conv = true;

    for(i=0;i<n;i++)
    {
        // Save current cluster of point i
        uint32_t curClust = dat[i].clusterID;

        // Compute WSS of datum i cluster without datum i
        uint32_t tmpClust = (curClust + 1 >= k) ? 0 : curClust + 1; // Define a tmp cluster

        CLUSTER_transferPointToCluster(dat, i, p, c, tmpClust); // Transfer datum i to tmp cluster

        // Update objects weights in former datum i cluster 
        if(internalObjectWeights == true)
        {
            // Internal computation of object weights
            CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, curClust, objWeiMet, dist);
        }

        //  Update features weights in former datum i cluster
        if(internalFeatureWeights == true)
        {
            // Internal computation of feature weights
            CLUSTER_computeFeatureWeightsInCluster(dat, n, p, c, k, curClust, feaWeiMet);
        }

        double tmpFromWss = CLUSTER_computeNkWeightedWSSInCluster(dat, n, p, c, k, curClust); // Compute from WSS

        double minSumWss = 1e20;
        uint32_t minK;
        double tmpToWss[k];
        bool improved = false;

        for(l=0;l<k;l++)
        {
            if(l != curClust)
            {
                double sumWssRef = wss[curClust] + wss[l];

                // Compute WSS of cluster l with datum i
                CLUSTER_transferPointToCluster(dat, i, p, c, l); // Transfer datum i to tmp cluster
                // Update objects weights in former datum i cluster 
                if(internalObjectWeights == true)
                {
                    // Internal computation of object weights
                    CLUSTER_computeObjectWeightsInCluster(dat, n, p, c, k, l, objWeiMet, dist);
                }

                // Update feature weights in cluster l
                if(internalFeatureWeights == true)
                {
                    // Internal computation of feature weights
                    CLUSTER_computeFeatureWeightsInCluster(dat, n, p, c, k, l, feaWeiMet);
                }

                tmpToWss[l] = CLUSTER_computeNkWeightedWSSInCluster(dat, n, p, c, k, l); // Compute to WSS

                double newWss = tmpFromWss + tmpToWss[l];

                // Test if the deplacement from curClust to l improves the sum of WSS
                if(newWss < sumWssRef)
                {
                    improved = true;

                    // Test if the new sum of WSS is minimal
                    if(newWss < minSumWss)
                    {
                        minSumWss = newWss;
                        minK = l;
                    }

                    // Reset convergence variable
                    *conv = false;
                }
            }
        }

        // Test if WSS improved
        if(improved == true)
        {
            CLUSTER_transferPointToCluster(dat, i, p, c, minK); // Transfer datum i to minK cluster
            wss[curClust] = tmpFromWss;
            wss[minK] = tmpToWss[minK];
        }
        else
        {
            // Transfer datum i to initial cluster
            CLUSTER_transferPointToCluster(dat, i, p, c, curClust); 
        }
    }
}

static double CLUSTER_computeSquaredDistancePointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d)
{
    double dist = 0.0;
    switch(d)
    {
        default:
        case DISTANCE_EUCLIDEAN:
            {
                uint64_t j;
                for(j=0;j<p;j++)
                {
                    double tmp = (dat->dim[j] - c->centroid[j]);
                    tmp *= tmp;

                    if(isnan(tmp))
                    {
                        dist += 0.0;
                    }
                    else
                    {
                        dist += tmp;
                    }
                }
            }
            break;
        case DISTANCE_OTHER:
            {
                dist = -1;
            }
            break;
    }

    return dist;
}

static double CLUSTER_computeSquaredFWDistancePointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d)
{
    double dist = 0.0;
    switch(d)
    {
        default:
        case DISTANCE_EUCLIDEAN:
            {
                uint64_t j;
                for(j=0;j<p;j++)
                {
                    double tmp = (dat->dim[j] - c->centroid[j]);
                    tmp *= tmp;
                    tmp *= c->fw[j];

                    if(isnan(tmp))
                    {
                        dist += 0.0;
                    }
                    else
                    {
                        dist += tmp;
                    }
                }
            }
            break;
        case DISTANCE_OTHER:
            {
                dist = -1;
            }
            break;
    }

    return dist;
}

static double CLUSTER_computeSquaredDistanceClusterToCluster(cluster *ci, cluster *cj, uint64_t p, eDistanceType d)
{
    double dist = 0.0;
    switch(d)
    {
        default:
        case DISTANCE_EUCLIDEAN:
            {
                uint64_t j;
                for(j=0;j<p;j++)
                {
                    double tmp = (ci->centroid[j] - cj->centroid[j]);
                    tmp *= tmp;

                    if(isnan(tmp))
                    {
                        dist += 0.0;
                    }
                    else
                    {
                        dist += tmp;
                    }
                }
            }
            break;
        case DISTANCE_OTHER:
            {
                dist = -1;
            }
            break;
    }

    return dist;
}

static double CLUSTER_computeSquaredDistanceWeightedPointToCluster(data *dat, uint64_t p, cluster *c, eDistanceType d, double *fw, double ow)
{
    if(!(dat == NULL || p < 1 || c == NULL || fw == NULL))
    {
        double dist = 0.0;
        switch(d)
        {
            default:
            case DISTANCE_EUCLIDEAN:
                {
                    uint64_t j;
                    for(j=0;j<p;j++)
                    {
                        double tmp = fw[j]*pow((dat->dim[j] - c->centroid[j]), 2.0); // Apply feature weights

                        if(isnan(tmp))
                        {
                            dist += 0.0;
                        }
                        else
                        {
                            dist += tmp;
                        }
                    }
                }
                break;
            case DISTANCE_OTHER:
                {
                    dist = -1;
                }
                break;
        }

        return (ow*dist); // Apply object weight
    }
    else
    {
        return -1.0;
    }
}

static void CLUSTER_computeCentroids(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint64_t i,j;
    uint32_t l;

    // Reset each centroid dimension to 0
    for(l=0;l<k;l++)
        for(j=0;j<p;j++)
            c[l].centroid[j] = 0.0;

    // Compute the new centroid dimension
    for(i=0;i<n;i++)
    {
        for(j=0;j<p;j++)
        {
            c[dat[i].clusterID].centroid[j] += (dat[i].dim[j]/(double)c[dat[i].clusterID].nbData);   
        }
    }    
}

static void CLUSTER_transferPointToCluster(data *dat, uint64_t indN, uint64_t p, cluster *c, uint32_t indK)
{
    uint64_t j;

    uint32_t prevClust = dat[indN].clusterID; // Retreive datum previous cluster

    // Remove point from former cluster
    CLUSTER_removePointFromCluster(&(dat[indN]), &(c[prevClust]));
    // Add point to cluster indK
    CLUSTER_addPointToCluster(&(dat[indN]), &(c[indK]));

    // Compute the new centroid dimensions
    for(j=0;j<p;j++)
    {
        if(c[prevClust].nbData == 0)
        {
            c[prevClust].centroid[j] = 0.0;
        }
        else
        {
            c[prevClust].centroid[j] = ((c[prevClust].centroid[j] * (double)(c[prevClust].nbData + 1)) - dat[indN].dim[j])/(double)c[prevClust].nbData;   
        }
        c[indK].centroid[j] = ((c[indK].centroid[j] * (double)(c[indK].nbData - 1)) + dat[indN].dim[j])/(double)c[indK].nbData;
    }
}

static void CLUSTER_addPointToCluster(data *dat, cluster *c)
{
    // Test if cluster is empty
    if(c->head == NULL)
    {
        dat->succ = NULL;
    }
    else
    {
        dat->succ = c->head;
        ((data *)c->head)->pred = dat;
    }

    dat->pred = NULL;
    c->head = dat;

    // Update data membership
    dat->clusterID = c->ind; 

    // Update number of data in the cluster
    c->nbData++;
}

static void CLUSTER_removePointFromCluster(data *dat, cluster *c)
{
    // Test if cluster is empty
    if(c->head == NULL)
    {
        // Nothing to do 
    }
    else
    {
        // Test if dat is the first element of the cluster chain list
        if(dat->pred == NULL)
        {
            c->head = dat->succ;

            //Test if there is only one element in the cluster chain list
            if(dat->succ != NULL)
            {
                ((data *)dat->succ)->pred = NULL;
            }
        }
        // Test if dat is the last element of the cluster chain list
        else if(dat->succ == NULL)
        {
            ((data *)dat->pred)->succ = dat->succ;
        }
        // Test if dat is somewhere in the cluster chain list
        else
        {
            ((data *)dat->pred)->succ = dat->succ;
            ((data *)dat->succ)->pred = dat->pred;
        }

        // Reset datum pred & succ 
        dat->pred = NULL;
        dat->succ = NULL; 

        // Update data membership
        dat->clusterID = -1;

        // Update number of data in the cluster
        c->nbData--;
    }
}

static double CLUSTER_computeSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist)
{
    if(dat == NULL || n < 2 || p < 1 || c == NULL || k < 2)
    {
        return -2.0;
    }
    else
    {
        double a[n], b[n], s[n], sk[k], distCluster[k];
        uint64_t i,j;
        uint32_t l;

        // Initilize sik
        for(l=0;l<k;l++) 
            sk[l] = 0.0;

        for(i=0;i<n;i++)
        {
            // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
            double d = 0.0;
            for(j=0;j<n;j++)
            {
                if((j != i) && (dat[j].clusterID == dat[i].clusterID))
                {
                    d += dist[i][j];
                }
            }

            if((c[dat[i].clusterID].nbData - 1) == 0)
            {
                a[i] = 0.0;
            }
            else
            {
                a[i] = d / (double)(c[dat[i].clusterID].nbData - 1);
            }

            // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
            for(l=0;l<k;l++)
            {
                distCluster[l] = 0.0;
            }

            for(j=0;j<n;j++)
            {
                if(dat[j].clusterID != dat[i].clusterID)
                {
                    distCluster[dat[j].clusterID] += (dist[i][j]/(double)c[dat[j].clusterID].nbData);
                }
            }

            b[i] = 1.0e20;
            for(l=0;l<k;l++)
            {
                if((l != dat[i].clusterID) && (distCluster[l] != 0) && (distCluster[l] < b[i]))
                {
                    b[i] = distCluster[l];
                }
            }

            // Calculate s[i]
            if(c[dat[i].clusterID].nbData == 1)
            {
                s[i] = 0;
            }
            else
            {
                s[i] = (b[i] != a[i]) ?  ((b[i] - a[i]) / fmax(a[i], b[i])) : 0.0;
            }
               
            sk[dat[i].clusterID]+= s[i] / (double)c[dat[i].clusterID].nbData;
        }

        double sil = 0.0;
        for(l=0;l<k;l++)
        {
            if(!isnan(sk[l]))
                sil += sk[l];
        }

        return (sil/k);
    }
}

static double CLUSTER_computeDistancePointToPoint(data *iDat, data *jDat, uint64_t p, eDistanceType d)
{
    if(iDat == NULL || jDat == NULL || p < 1)
    {
        return -1.0;
    }
    else
    {
        double dist = 0;
        switch(d)
        {
            default:
            case DISTANCE_EUCLIDEAN:
                {
                    uint64_t j;
                    for(j=0;j<p;j++)
                        dist += pow((iDat->dim[j] - jDat->dim[j]), 2.0);
                }
                break;
            case DISTANCE_OTHER:
                {
                    dist = -1;
                }
                break;
        }

        return sqrt(dist);
    }
}

static double CLUSTER_computeCH(data *dat, cluster *c, double SSE, uint64_t n, uint64_t p, uint32_t k)
{
    uint64_t i, j;
    uint32_t l;
    double mean[p];
    for(j=0;j<p;j++)
    {
        mean[j] = 0.0;
        for(i=0;i<n;i++)
        {
            mean[j] += (double) dat[i].dim[j] / (double) n;
        }
    }

    double SSB = 0.0;

    for(l=0;l<k;l++)
    {
        for(j=0;j<p;j++)
        {
            SSB += (double) c[l].nbData * pow((c[l].centroid[j] - mean[j]), 2.0); 
        }
    }

    return (SSB / (k - 1)) / (SSE / (n - k)); 
}

void CLUSTER_initObjectWeights(data *dat, uint64_t n)
{
    uint64_t i;
    for(i=0;i<n;i++)
        dat[i].ow = 1.0;
}

static void CLUSTER_computeFeatureWeights(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, eMethodType m)
{
    switch(m)
    {
        default:
        case METHOD_DISPERSION :
            {
                CLUSTER_computeFeatureWeightsViaDispersion(dat, n, p, c, k, 2); // Using L2-norm
            }
            break;
        case METHOD_OTHER:
            {
                // Nothing to do
            }
            break;
    }
}

static void CLUSTER_computeFeatureWeightsInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, eMethodType m)
{
    switch(m)
    {
        default:
        case METHOD_DISPERSION :
            {
                CLUSTER_computeFeatureWeightsInClusterViaDispersion(dat, n, p, c, k, indK, 2); // Using L2-norm
            }
            break;
        case METHOD_OTHER:
            {
               // Nothing to do 
            }
            break;
    }
}

static void CLUSTER_computeFeatureWeightsInClusterViaDispersion(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, uint8_t norm)
{
    uint64_t i, j, m;
    double disp[p]; // Dispersion per feature
    uint64_t nbdataClust = c[indK].nbData;

    // Initialization
    for(j=0;j<p;j++)
    {
        disp[j] = 0.0;
    }

    // Compute dispersion
    data *pti = (data *)c[indK].head;
    for(i=0;i<nbdataClust;i++)
    {
        for(j=0;j<p;j++)
        {
            disp[j] += CLUSTER_computeFeatureDispersion(pti, j, &(c[indK]), norm);
        }

        // Update pti
        pti = (data *)pti->succ;
    }

    // Compute weights
    double tmp[p];
    for(j=0;j<p;j++)
    {
        tmp[j] = 0.0;
    }

    for(j=0;j<p;j++)
    {
        for(m=0;m<p;m++)
        {
            tmp[j] += pow((disp[j] / disp[m]),(1 / (norm - 1)));
            if(isnan(tmp[j]))
                tmp[j] = 0.0; 
        }

        if(c[indK].nbData == 1)
        {
            c[indK].fw[j] = 1.0;
        }
        else
        {
            // The sum of features weights as to be equal to unity 
            c[indK].fw[j] = pow((1 / tmp[j]), norm);
            if(isinf(c[indK].fw[j]))
                c[indK].fw[j] = 1.0; 
        }
    }
}

static void CLUSTER_computeFeatureWeightsViaDispersion(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint8_t norm)
{
    uint64_t i, j;
    uint32_t l,m;
    double disp[k][p]; // Dispersion per cluster and per feature

    // Initialization
    for(l=0;l<k;l++)
    {
        for(j=0;j<p;j++)
        {
            disp[l][j] = 0.0;
        }
    }

    // Compute dispersion
    for(i=0;i<n;i++)
    {
        for(j=0;j<p;j++)
        {
            disp[dat[i].clusterID][j] += CLUSTER_computeFeatureDispersion(&(dat[i]), j, &(c[dat[i].clusterID]), norm);
        }
    }

    // Compute weights
    for(l=0;l<k;l++)
    {
        double tmp[p];
        for(j=0;j<p;j++)
        {
            tmp[j] = 0.0;
        }

        for(j=0;j<p;j++)
        {
            for(m=0;m<p;m++)
            {
                tmp[j] += pow((disp[l][j] / disp[l][m]),(1 / (norm - 1)));
                if(isnan(tmp[j]))
                    tmp[j] = 0.0; 
            }

            if(c[l].nbData == 1)
            {
                //fw[l][j] = 1.0;
                c[l].fw[j] = 1.0;
            }
            else
            {
                // The sum of features weights as to be equal to unity 
                c[l].fw[j] = pow((1 / tmp[j]), norm);

                if(isinf(c[l].fw[j]))
                    c[l].fw[j] = 1.0;
            }
        }
    }
}

static double CLUSTER_computeFeatureDispersion(data *dat, uint64_t p, cluster *c, uint8_t norm)
{
    return pow(dat->dim[p] - c->centroid[p], norm); 
}

static void CLUSTER_computeObjectWeights(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, eMethodType m, double **dist)
{
    switch(m)
    {
        case METHOD_SILHOUETTE :
            {
                CLUSTER_computeObjectWeightsViaSilhouette(dat, n, p, c, k, dist);
            }
            break;
        case METHOD_SILHOUETTE_NK :
            {
                CLUSTER_computeObjectWeightsViaSilhouetteNK(dat, n, p, c, k, dist);
            }
            break;
        case METHOD_MEDIAN :
            {
                CLUSTER_computeObjectWeightsViaMedian(dat, n, p, c, k);
            }
            break;
        case METHOD_MEDIAN_NK :
            {
                CLUSTER_computeObjectWeightsViaMedianNK(dat, n, p, c, k);
            }
            break;
        case METHOD_MIN_DIST_CENTROID :
            {
                CLUSTER_computeObjectWeightsViaMinDistCentroid(dat, n, p, c, k);
            }
            break;
        case METHOD_MIN_DIST_CENTROID_NK :
            {
                CLUSTER_computeObjectWeightsViaMinDistCentroidNK(dat, n, p, c, k);
            }
            break;
        case METHOD_SUM_DIST_CENTROID :
            {
                CLUSTER_computeObjectWeightsViaSumDistCentroid(dat, n, p, c, k);
            }
            break;
        default:
            {
                // Nothing to do
            }
            break;
    }
}

static void CLUSTER_computeObjectWeightsInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, eMethodType m, double **dist)
{
    switch(m)
    {
        case METHOD_SILHOUETTE :
            {
                CLUSTER_computeObjectWeightsInClusterViaSilhouette(dat, n, p, c, k, indK, dist);
            }
            break;
        case METHOD_SILHOUETTE_NK :
            {
                CLUSTER_computeObjectWeightsInClusterViaSilhouetteNK(dat, n, p, c, k, indK, dist);
            }
            break;
        case METHOD_MEDIAN :
            {
                CLUSTER_computeObjectWeightsInClusterViaMedian(dat, n, p, c, indK);
            }
            break;
        case METHOD_MEDIAN_NK :
            {
                CLUSTER_computeObjectWeightsInClusterViaMedianNK(dat, n, p, c, indK);
            }
            break;
        case METHOD_MIN_DIST_CENTROID :
            {
                CLUSTER_computeObjectWeightsInClusterViaMinDistCentroid(dat, n, p, c, k, indK);
            }
            break;
        case METHOD_MIN_DIST_CENTROID_NK :
            {
                CLUSTER_computeObjectWeightsInClusterViaMinDistCentroidNK(dat, n, p, c, k, indK);
            }
            break;
        case METHOD_SUM_DIST_CENTROID :
            {
                CLUSTER_computeObjectWeightsInClusterViaSumDistCentroid(dat, n, p, c, k, indK);
            }
            break;
        default:
            {
                // Nothing to do
            }
            break;
    }
}

static void CLUSTER_computeObjectWeightsViaSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist)
{
    uint32_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaSilhouette(dat, n, p, c, k, l, dist);
    }
}

static void CLUSTER_computeObjectWeightsViaSilhouetteNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double **dist)
{
    uint32_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaSilhouetteNK(dat, n, p, c, k, l, dist);
    }
}

static void CLUSTER_computeObjectWeightsInClusterViaSilhouette(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double **dist)
{
    uint64_t nbdataClust = c[indK].nbData;
    double a[nbdataClust], b[nbdataClust], s[nbdataClust], sk = 0.0, distCluster[k];
    uint64_t i,j;
    uint32_t l;

    data *pti = (data *)c[indK].head;
    for(i=0;i<nbdataClust;i++)
    {
        // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
        double d = 0.0;
        data *ptj = (data *)c[indK].head;
        for(j=0;j<nbdataClust;j++)
        {
            if(ptj->ind != pti->ind)
            {
                d += dist[pti->ind][ptj->ind];
            }

            // Update ptj
            ptj = (data *)ptj->succ;
        }

        if((c[indK].nbData - 1) == 0)
        {
            a[i] = 0.0;
        }
        else
        {
            a[i] = d / (double)(c[indK].nbData - 1);
        }

        // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
        for(l=0;l<k;l++)
            distCluster[l] = 0.0;

        for(j=0;j<n;j++)
            if(dat[j].clusterID != indK)
                distCluster[dat[j].clusterID] += (dist[pti->ind][j]/c[dat[j].clusterID].nbData);
        b[i] = 1.0e20;
        for(l=0;l<k;l++)
            if(l != indK && distCluster[l] != 0 && distCluster[l] < b[i])
                b[i] = distCluster[l];

        // Calculate s[i]
        if(c[indK].nbData == 1)
        {
            s[i] = 0;
        }
        else
        {
            s[i] = (b[i] != a[i]) ?  ((b[i] - a[i]) / fmax(a[i], b[i])) : 0.0;
        }

        if(s[i] < 0 || s[i] > 0)
            s[i] = 1 - ((s[i]+1)/2); // Rescale silhouette to 0-1 
        else // si = 0
            s[i] = 0.5;

        // Calculate sum of s[i] per cluster 
        sk += s[i];

        pti->ow = s[i];

        if(isnan(pti->ow) || isinf(pti->ow))
        {
            pti->ow = 1.0;
        }

        // Update pti
        pti = (data *)pti->succ;
    }
}

static void CLUSTER_computeObjectWeightsInClusterViaSilhouetteNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK, double **dist)
{
    uint64_t nbdataClust = c[indK].nbData;
    double a[nbdataClust], b[nbdataClust], s[nbdataClust], sk = 0.0, distCluster[k];
    uint64_t i,j;
    uint32_t l;

    data *pti = (data *)c[indK].head;
    for(i=0;i<nbdataClust;i++)
    {
        // Calculate a[i], the average dissimilarity of i with all other data within the same cluster
        double d = 0.0;
        data *ptj = (data *)c[indK].head;
        for(j=0;j<nbdataClust;j++)
        {
            if(ptj->ind != pti->ind)
            {
                d += dist[pti->ind][ptj->ind];
            }

            // Update ptj
            ptj = (data *)ptj->succ;
        }

        if((c[indK].nbData - 1) == 0)
        {
            a[i] = 0.0;
        }
        else
        {
            a[i] = d / (double)(c[indK].nbData - 1);
        }

        // Calculate b[i], the lowest average dissimilarity of i to any other cluster, of which i is not a member
        for(l=0;l<k;l++)
            distCluster[l] = 0.0;

        for(j=0;j<n;j++)
            if(dat[j].clusterID != indK)
                distCluster[dat[j].clusterID] += (dist[pti->ind][j]/c[dat[j].clusterID].nbData);
        b[i] = 1.0e20;
        for(l=0;l<k;l++)
            if(l != indK && distCluster[l] != 0 && distCluster[l] < b[i])
                b[i] = distCluster[l];

        // Calculate s[i]
        if(c[indK].nbData == 1)
        {
            s[i] = 0;
        }
        else
        {
            s[i] = (b[i] != a[i]) ?  ((b[i] - a[i]) / fmax(a[i], b[i])) : 0.0;
        }

        if(s[i] < 0 || s[i] > 0)
            s[i] = 1 - ((s[i]+1)/2); // Rescale silhouette to 0-1 
        else // si = 0
            s[i] = 0.5;

        // Calculate sum of s[i] per cluster 
        sk += s[i];

        // Update pti
        pti = (data *)pti->succ;
    }

    // Calculate object weights
    pti = (data *)c[indK].head;
    for(i=0;i<nbdataClust;i++)
    {
        pti->ow = (s[i] / sk) * (double) c[indK].nbData;

        if(isnan(pti->ow) || isinf(pti->ow))
        {
            pti->ow = 1.0;
        }
        // Update pti
        pti = (data *)pti->succ;
    }
}

static void CLUSTER_computeObjectWeightsViaMedian(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint32_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaMedian(dat, n, p, c, l);
    }
}

static void CLUSTER_computeObjectWeightsViaMedianNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint32_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaMedianNK(dat, n, p, c, l);
    }
}

static int cmpfunc(const void * a, const void * b)
{
    return (*(double*)a > *(double*)b) ? 1 : (*(double*)a < *(double*)b) ? -1:0 ;
}

static void CLUSTER_computeObjectWeightsInClusterViaMedian(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t indK)
{
    uint64_t i, j;
    uint64_t nbDataClust = c[indK].nbData;
    double median[p]; // Median per cluster
    double w[nbDataClust]; // Tmp weights
    double sumWeights = 0.0; // Sum of weights in cluster k

    for(j=0;j<p;j++)
    {
        double dim[nbDataClust];
        data *pti = (data *)c[indK].head;
        for(i=0;i<nbDataClust;i++)
        {
            dim[i] = pti->dim[j];

            // Update pti
            pti = (data *)pti->succ;
        }

        // Sort dimension value in ascending way
        qsort(dim, nbDataClust, sizeof(double), cmpfunc);

        // Compute the median
        if(!(nbDataClust % 2))
        {
            median[j] = (dim[(nbDataClust / 2)] + dim[(nbDataClust / 2) - 1]) / 2;
        }
        else
        {
            median[j] = dim[((nbDataClust + 1) / 2) - 1];
        }
    }

    // Compute tmp weights
    data *pti = (data *)c[indK].head;
    for(i=0;i<nbDataClust;i++)
    {
        // Initialize tmp weights
        w[i] = 0.0;

        for(j=0;j<p;j++)
        {
            w[i] += fabs(pti->dim[j] - median[j]); 
        }

        sumWeights += w[i];

        if(nbDataClust == 1)
        {
            pti->ow = 1.0;
        }
        else
        {
            pti->ow = w[i];
        }

        // Update pti
        pti = (data *)pti->succ;
    }
}

static void CLUSTER_computeObjectWeightsInClusterViaMedianNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t indK)
{
    uint64_t i, j;
    uint64_t nbDataClust = c[indK].nbData;
    double median[p]; // Median per cluster
    double w[nbDataClust]; // Tmp weights
    double sumWeights = 0.0; // Sum of weights in cluster k

    for(j=0;j<p;j++)
    {
        double dim[nbDataClust];
        data *pti = (data *)c[indK].head;
        for(i=0;i<nbDataClust;i++)
        {
            dim[i] = pti->dim[j];

            // Update pti
            pti = (data *)pti->succ;
        }

        // Sort dimension value in ascending way
        qsort(dim, nbDataClust, sizeof(double), cmpfunc);

        // Compute the median
        if(!(nbDataClust % 2))
        {
            median[j] = (dim[(nbDataClust / 2)] + dim[(nbDataClust / 2) - 1]) / 2;
        }
        else
        {
            median[j] = dim[((nbDataClust + 1) / 2) - 1];
        }
    }

    // Compute tmp weights
    data *pti = (data *)c[indK].head;
    for(i=0;i<nbDataClust;i++)
    {
        // Initialize tmp weights
        w[i] = 0.0;

        for(j=0;j<p;j++)
        {
            w[i] += fabs(pti->dim[j] - median[j]); 
        }

        sumWeights += w[i];

        // Update pti
        pti = (data *)pti->succ;
    }

    // Compute objects weights
    pti = (data *)c[indK].head;
    for(i=0;i<nbDataClust;i++)
    {
        if(nbDataClust == 1)
        {
            pti->ow = 1.0;
        }
        else
        {
            pti->ow = (w[i] / sumWeights) * (double) nbDataClust;
        }

        // Update pti
        pti = (data *)pti->succ;
    }
}

static void CLUSTER_computeObjectWeightsViaMinDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint64_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaMinDistCentroid(dat, n, p, c, k, l);
    }
}

static void CLUSTER_computeObjectWeightsViaMinDistCentroidNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint64_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaMinDistCentroidNK(dat, n, p, c, k, l);
    }
}

static void CLUSTER_computeObjectWeightsViaSumDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    uint64_t l;

    for(l=0;l<k;l++)
    {
        CLUSTER_computeObjectWeightsInClusterViaSumDistCentroid(dat, n, p, c, k, l);
    }
}

static void CLUSTER_computeObjectWeightsInClusterViaMinDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK)
{
    // Based on the nearest centroid different of the point one

    uint64_t i;
    uint64_t l;
    uint64_t nbDataClust = c[indK].nbData;
    double w[nbDataClust]; // Tmp weights
    double sumWeights = 0.0; // Sum of weights in cluster k

    // Compute tmp weights
    data *pti = (data *)c[indK].head;
    for(i=0;i<nbDataClust;i++)
    {
        // Initialize tmp weights
        w[i] = 1e20;

        for(l=0;l<k;l++)
        {
            if(l != indK)
            {
                // Calculate squared Euclidean distance
                double dist = CLUSTER_computeSquaredDistancePointToCluster(pti, p, &(c[l]), DISTANCE_EUCLIDEAN);

                if(dist < w[i])
                {
                    w[i] = dist;
                }
            }
        } 

        // Compute ratio distance with its centroid / distance with the nearest other centroid 
        w[i] = 1.0 / w[i];
        sumWeights += w[i];

        if(nbDataClust == 1)
        {
            pti->ow = 1.0;
        }
        else
        {
            pti->ow = w[i];
        }

        // Update pti
        pti = (data *)pti->succ;
    }
}

static void CLUSTER_computeObjectWeightsInClusterViaMinDistCentroidNK(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK)
{
    // Based on the nearest centroid different of the point one

    uint64_t i;
    uint64_t l;
    uint64_t nbDataClust = c[indK].nbData;
    double w[nbDataClust]; // Tmp weights
    double sumWeights = 0.0; // Sum of weights in cluster k

    // Compute tmp weights
    data *pti = (data *)c[indK].head;
    for(i=0;i<nbDataClust;i++)
    {
        // Initialize tmp weights
        w[i] = 1e20;

        for(l=0;l<k;l++)
        {
            if(l != indK)
            {
                // Calculate squared Euclidean distance
                double dist = CLUSTER_computeSquaredDistancePointToCluster(pti, p, &(c[l]), DISTANCE_EUCLIDEAN);

                if(dist < w[i])
                {
                    w[i] = dist;
                }
            }
        } 

        // Compute ratio distance with its centroid / distance with the nearest other centroid 
        w[i] = 1.0 / w[i];
        sumWeights += w[i];

        // Update pti
        pti = (data *)pti->succ;
    }

    // Compute objects weights
    pti = (data *)c[indK].head;
    for(i=0;i<nbDataClust;i++)
    {
        if(nbDataClust == 1)
        {
            pti->ow = 1.0;
        }
        else
        {
            pti->ow = (w[i] / sumWeights) * (double) nbDataClust;
        }

        // Update pti
        pti = (data *)pti->succ;
    }
}

static void CLUSTER_computeObjectWeightsInClusterViaSumDistCentroid(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK)
{
    // Based on the sum of distances with other centroids 

    uint64_t i;
    uint64_t l;
    uint64_t nbDataClust = c[indK].nbData;
    double sumDist = 0.0;

    // Compute sum of distances with other centroids
    for(l=0;l<k;l++)
    {
        if(l != indK)
        {
            // Calculate squared Euclidean distance
            sumDist += CLUSTER_computeSquaredDistanceClusterToCluster(&(c[indK]), &(c[l]), p, DISTANCE_EUCLIDEAN);

        }
    }

    // Compute objects weights
    data *pti = (data *)c[indK].head;
    double sumWei = 0.0;
    for(i=0;i<nbDataClust;i++)
    {
        pti->ow = (1 / sumDist);

        sumWei += pti->ow;

        // Update pti
        pti = (data *)pti->succ;
    }
}

static double CLUSTER_computeSSE(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k)
{
    double SSE = 0.0;
    uint64_t i;

    for(i=0;i<n;i++)
    {
        SSE += CLUSTER_computeSquaredDistancePointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN); 
    }

    return SSE;
}

static void CLUSTER_computeNkWeightedWSS(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, double wss[k])
{
    uint64_t i;
    uint32_t l;

    for(l=0;l<k;l++)
    {
        wss[l] = 0.0;
    }

    for(i=0;i<n;i++)
    {
        wss[dat[i].clusterID] += CLUSTER_computeSquaredDistanceWeightedPointToCluster(&(dat[i]), p, &(c[dat[i].clusterID]), DISTANCE_EUCLIDEAN, (double *)c[dat[i].clusterID].fw, dat[i].ow);
    }
}

static double CLUSTER_computeNkWeightedWSSInCluster(data *dat, uint64_t n, uint64_t p, cluster *c, uint32_t k, uint32_t indK)
{
    uint64_t i;
    double wss = 0.0;

    data *pti = (data *)c[indK].head;  
    for(i=0;i<c[indK].nbData;i++)
    {
        wss += CLUSTER_computeSquaredDistanceWeightedPointToCluster(pti, p, &(c[indK]), DISTANCE_EUCLIDEAN, (double *)c[indK].fw, pti->ow);

        // Update pt
        pti = (data *)pti->succ;
    }

    return wss;
}

void CLUSTER_ComputeMatDistPointToPoint(data *dat, uint64_t n, uint64_t p, double ***dist)
{
    if(dat == NULL || n < 2 || p < 1)
    {
        // Nothing to do
    }
    else
    {
        *dist = malloc(n * sizeof(double *));
        if(*dist == NULL)
        {
            // Nothing to do
        }
        else
        {
            uint64_t i, j;
            // Create the distance matrix of i vs j
            for(i=0;i<n;i++)
            {
                (*dist)[i] = malloc(n * sizeof(double));
                if((*dist)[i] == NULL)
                {
                    // Nothing to do
                }
                else
                {
                    for (j=0;j<n;j++)
                    {
                        (*dist)[i][j]= CLUSTER_computeDistancePointToPoint(&(dat[i]), &(dat[j]), p, DISTANCE_EUCLIDEAN);
                    }
                }
            }
        }
    }
}

void CLUSTER_FreeMatDistPointToPoint(uint64_t n, double ***dist)
{
    if(n < 2)
    {
        // Nothing to do
    }
    else
    {
        uint64_t i;
        for (i=0;i<n;i++)
        {
            free((*dist)[i]);
        }
        free(*dist);
    }
}
