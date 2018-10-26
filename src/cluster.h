/** @file cluster.h
 *  @brief Function prototypes for the k-means clustering.
 *
 *  This contains the function prototypes for the k-means
 *  clustering and eventually any macros, constants, or 
 *  global variables.
 *
 *  @author Alexandre Gondeau
 *  @bug No known bugs.
 */

#ifndef _CLUSTER_H
#define _CLUSTER_H

/* -- Includes -- */

/* libc includes. */
#include <stdint.h>
#include <stdbool.h>

/** @brief Contains clusters informations.
 *
 */
typedef struct _cluster
{
    uint32_t ind; // Cluster index
    double *centroid; // Centroid dimensions 
    uint64_t nbData; // Number of data in cluster
    double *fw; // Features weights in cluster
    void *head; // Pointer to the head of cluster chained list
} cluster;

/** @brief Contains data informations.
 *
 */
typedef struct _data
{
    uint64_t ind; // Data index
    double *dim; // Data dimensions
    uint32_t clusterID; // Cluster ID of data
    double ow; // Data weight
    void *pred; // Pointer to datum predecessor in cluster chained list
    void *succ; // Pointer to datum successor in cluster chained list
} data;

/** @brief Contains the different weights calculation methods.
 *
 */
typedef enum _eMethodType 
{
    METHOD_SILHOUETTE = 0,
    METHOD_SILHOUETTE_NK,
    METHOD_MEDIAN,
    METHOD_MEDIAN_NK,
    METHOD_MIN_DIST_CENTROID,
    METHOD_MIN_DIST_CENTROID_NK,
    METHOD_SUM_DIST_CENTROID,
    METHOD_DISPERSION,
    METHOD_OTHER
} eMethodType; 

/** @brief Allocates memory for the clusters dimensions.  
 *
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @param cen The pointer to the chosen data number as centroid. 
 *  @param dat The pointer to data.
 *  @return Void.
 */
void CLUSTER_initClusters(uint64_t p, cluster *c, uint32_t k, int32_t *cen, data *dat);

/** @brief Frees allocated memory for the clusters dimensions.  
 *
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return Void.
 */
void CLUSTER_freeClusters(cluster *c, uint32_t k);

/** @brief Computes the classical version of k-means  
 *         algorithm for k clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param c The pointer to the clusters.
 *  @param k The number of clusters. 
 *  @return The sum of squared errors for the clustering.
 */
double CLUSTER_kmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c);

/** @brief Computes the features 
 * version of k-means algorithm for k clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param internalFeatureWeights The boolean that 
 *         specified if the features weights come 
 *         from internal computation or from a file.
 *  @param featureWeightsMethod The feature weights calculation
 *         method.         
 *  @return The sum of squared errors for the clustering.
 */
double CLUSTER_featuresWeightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, eMethodType featureWeightsMethod);

/** @brief Computes the objects (and features) 
 * version of k-means algorithm for k clusters.
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param k The number of clusters. 
 *  @param c The pointer to the clusters.
 *  @param internalFeatureWeights The boolean specifying if 
 *              the features weights are computed internally.
 *  @param  feaWeiMet The method used to computed the features
 *                    weights internally.
 *  @param internalObjectsWeights The boolean specifying if 
 *              the objects weights are computed internally.
 *  @param  objWeiMet The method used to computed the objects
 *                    weights internally.
 *  @param dist The double pointer to the distance matrix point to point.
 *  @return The sum of squared errors for the clustering.
 */
double CLUSTER_weightedKmeans(data *dat, uint64_t n, uint64_t p, uint32_t k, cluster *c, bool internalFeatureWeights, eMethodType featureWeightsMethod, bool internalObjectWeights, eMethodType objectWeightsMethod, double **dist);

/** @brief Initializes objects weights to 1. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of objects. 
 *  @return Void.
 */
void CLUSTER_initObjectWeights(data *dat, uint64_t n);

/** @brief Computes the matrix of distances 
 *         points to points. 
 *
 *  @param dat The pointer to data.
 *  @param n The number of the data.
 *  @param p The number of data dimensions.
 *  @param dist The triple pointer to the matrix of distances.
 *  @return Void.
 */
void CLUSTER_ComputeMatDistPointToPoint(data *dat, uint64_t n, uint64_t p, double ***dist);

/** @brief Frees the matrix of distances 
 *         points to points. 
 *
 *  @param n The number of the data.
 *  @param dist The triple pointer to the matrix of distances.
 *  @return Void.
 */
void CLUSTER_FreeMatDistPointToPoint(uint64_t n, double ***dist);

#endif /* _CLUSTER_H */
