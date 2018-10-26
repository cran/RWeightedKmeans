#' Classical k-means computation
#'
#' This function allows to compute a classical version of k-means algorithm (Lloyd).
#' @param dat Data to cluster.
#' @param k The clustering is processed for k partitions.
#' @param nbRep Specify the number of random starts.
#' @keywords Lloyd kmeans
#' @export
#' @examples
#' kmeans()

kmeans <- function(dat, k=2, nbRep=100)
{
    n <- as.integer(nrow(dat))
    p <- as.integer(ncol(dat))

    # Test args
    if(is.na(n) || is.na(p)) stop("Invalid input data")
    if(is.na(k) || k < 2 || k > (n-1)) stop("Invalid value of group number")
    if(is.na(nbRep) || nbRep < 1) stop("Invalid value of repetition number")

    # Choose k data ID as centroids
    cen <- sample.int(n, k, replace=FALSE)

    # Compute k-means algorithm
    cl <- .Call("C_kmeans", as.double(dat), as.integer(cen), n, p, as.integer(k))

    # Compute clustering quality indices
    bestCH <- calinhara(dat,cl,cn=k) # Calinski-Harabasz
    bestSil <- as.numeric(intCriteria(dat, cl, "Silhouette")) # Silhouette
    clCH <- cl
    clSil <- cl
    if(nbRep >= 2L )
        for(i in 2:nbRep) 
        {
            # Choose k data ID as centroids
            cen <- sample.int(as.integer(dim(dat)[1]), k, replace=FALSE)

            # Compute k-means algorithm
            cl <- .Call("C_kmeans", as.double(dat), as.integer(cen), n, p, as.integer(k))

            # Compute clustering quality indices
            ch <- calinhara(dat, cl, cn=k) # Calinski-Harabasz
            sil <- as.numeric(intCriteria(dat, cl, "Silhouette")) # Silhouette
            if(ch > bestCH && length(unique(cl)) == k) 
            {
                clCH <- cl
                bestCH <- ch
            }
            if(sil > bestSil && length(unique(cl)) == k) 
            {
                clSil <- cl
                bestSil <- sil
            }
        }

    structure(list(k = k, bestCH = bestCH, clusteringCH = clCH, bestSil = bestSil, clusteringSil = clSil, algorithm = "Classical k-means"))
}

#' Weighted k-means computation
#'
#' This function allows to compute a weighted version of k-means algorithm.
#' @param dat Data to cluster.
#' @param k The clustering is processed for k partitions.
#' @param nbRep Specify the number of random starts.
#' @param ifc Specify the algorithm needs to use an internal computation of feature weights.
#' @param ioc Specify the algorithm needs to use an internal computation of object weights.
#' @param fwm Specify the features weights calculation method.
#' @param ioc Specify the objects weights calculation method.
#' @param owm Specify the objects weights calculation method.
#' @keywords weighted kmeans
#' @export
#' @examples
#' weightedKmeans()

weightedKmeans <- function(dat, k=2, nbRep=100, ifc=FALSE, ioc=TRUE, fwm="DISP", owm="SIL")
{
    n <- as.integer(nrow(dat))
    p <- as.integer(ncol(dat))

    # Test args
    if(is.na(n) || is.na(p)) stop("Invalid input data")
    if(is.na(k) || k < 2 || k > (n-1)) stop("Invalid value of group number")
    if(is.na(nbRep) || nbRep < 1) stop("Invalid value of repetition number")
    if(is.na(ifc)) stop("Invalid value for ifc parameter")
    if(is.na(ioc)) stop("Invalid value for ioc parameter")
    if(is.na(fwm)) stop("Invalid value for fwm parameter")
    if(is.na(owm)) stop("Invalid value for owm parameter")

    # Choose k data ID as centroids
    cen <- sample.int(n, k, replace=FALSE)

    # Compute weighted k-means algorithm
    cl <- .Call("C_weightedKmeans", as.double(dat), as.integer(cen), n, p, as.integer(k), as.logical(ifc), as.logical(ioc), as.character(fwm), as.character(owm))

    clust <- cl[[1]]
    ow <- cl[[2]]

    # Compute clustering quality indices
    bestCH <- calinhara(dat, clust, cn=k) # Calinski-Harabasz
    bestSil <- as.numeric(intCriteria(dat, clust, "Silhouette")) # Silhouette
    if(!is.na(bestCH)) bestCH <- 0
    if(!is.na(bestSil)) bestSil <- -1
    clCH <- clust
    clSil <- clust
    bestOwCH <- ow
    bestOwSil <- ow
    if(nbRep >= 2L )
        for(i in 2:nbRep) 
        {
            # Choose k data ID as centroids
            cen <- sample.int(as.integer(dim(dat)[1]), k, replace=FALSE)

            # Compute weighted k-means algorithm
            cl <- .Call("C_weightedKmeans", as.double(dat), as.integer(cen), n, p, as.integer(k), as.logical(ifc), as.logical(ioc), as.character(fwm), as.character(owm))

            clust <- cl[[1]]
            ow <- cl[[2]]

            # Compute clustering quality indices
            ch <- calinhara(dat, clust, cn=k) # Calinski-Harabasz
            sil <- as.numeric(intCriteria(dat, clust, "Silhouette")) # Silhouette

            if(!is.na(ch))
            {
                if(ch > bestCH && length(unique(clust)) == k) 
                {
                    clCH <- clust
                    bestOwCH <- ow
                    bestCH <- ch
                }
            }
            if(!is.na(sil))
            {
                if(sil > bestSil && length(unique(clust)) == k) 
                {
                    clSil <- clust
                    bestSil <- sil
                    bestOwSil <- ow
                }
            }
        }

    structure(list(k = k, bestCH = bestCH, clusteringCH = clCH, objectWeightCH = bestOwCH, bestSil = bestSil, clusteringSil = clSil, objectWeightSil = bestOwSil, algorithm = "Weighted k-means"))
}
