
class clusters(object):

    def __init__(self, ):
        self._clusters = []

    ### change this algorithm to find the cluster
    ### with a maximum abount of overlap above a
    ### threshold.
    def _findClusterFor(self, obj, sim_pred, thresh):
        maxcluster = None
        maxsim = 0.0
        for c in self._clusters:
            for objc in c:
                if obj == objc:
                    continue
                sim = sim_pred(obj, objc)
                if sim >= thresh:
                    if sim > maxsim:
                        maxsim = sim
                        maxcluster = c
        #print('obj        = {}'.format(obj))
        #print('maxcluster = {}'.format(maxcluster))
        #print('maxsim     = {}'.format(maxsim))
        return maxcluster

    def _makeCluster(self, obj):
        self._clusters.append([obj])

    def cluster(self, obj, sim_pred, thresh):
        clr = self._findClusterFor(obj, sim_pred, thresh)
        if clr is None:
            self._makeCluster(obj)
        else:
            clr.append(obj)

    def getClusters(self):
        return self._clusters

    def clearClusters(self):
        self._clusters = []

    def addCluster(self, cl):
        self._clusters.append(cl)

    


    

