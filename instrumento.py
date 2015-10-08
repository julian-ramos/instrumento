import dataSearchUtils as dS
import datetime as dt
import os
import os.path as op
import traceback



class instrumento:
    """
    The instrumento class:
    
    This class helps to document easily a coding process. This class is specifically 
    aimed at Machine learning/Data mining and data analysis tasks. Here is how you can use it
    """
    
    
    def __init__(self,**kwargs):
        self.printout=False
        self.activeLog="/Users/ingenia/git/instrumento/activeLog.txt"
        
        """
        arguments are
        path : Path to where the log is going to be stored
        logname : Filename for the log
        
        This function does not check whether the file exists or not
        if it exists it will append to the current content
        """
        
#         if "path" not in kwargs.keys() or "logname" not in kwargs.keys():
#             print("either path ")
#             return
        
        if "path" in kwargs.keys() and "logname" in kwargs.keys() :
            self.path=kwargs['path']
            self.logname=kwargs['logname']
            file=open(self.activeLog,"w")
            file.write("%s \n"%(self.path))
            file.write("%s"%(self.logname))
            file.close()
        else:
            try:
                file=open(self.activeLog,"r")
#                 print(file.readline())
#                 print(file.readline())
                temp1=file.readline()
                temp2=file.readline()
                file.close()
            except:
                raise NameError("Could not read current activeLog")
            self.logname=temp2.strip()
            self.path=temp1.strip()
            print(temp1,temp2)
                   
        self.filename=self.path+'/'+self.logname
        if 'printout' in kwargs.keys():
            self.printout=kwargs['printout']


    def printlog(self,text):
        if self.printout:
            print(text)
            
    def act(self,activity):
            """
            Create a log of the activity you are currently working on
            in a segment of code
            """
            file=open(self.filename,"a")
            
            text="\n"+"activity,"+activity+","+dt.datetime.now().strftime("%b-%d-%I:%M:%S%p")
            file.write(text)
            file.close()
            self.printlog(text)
            
    def sum(self,summary, description=""):
            """
            Create a summary log of the current activity, for instance if working with an ML algorithm
            here it can be stored the accuracy, f1-score or other measures that are of importance
            to evaluate performance. Here, you can also store the output from a feature selection process
            or metrics from a preprocessing step.
            """
            file=open(self.filename,"a")
            
            text="\n"+"summary,"+str(summary)+' '+description+","+dt.datetime.now().strftime("%b-%d-%I:%M:%S%p")
            file.write(text)
            file.close()
            self.printlog(text)
            
    def params(self,*args):
            """
            Create a log of the parameters used with your code. 
            Examples of what parameter is are:
            - kernel of an SVM
            - Features used
            - Specific training data set
            """
            stack = traceback.extract_stack()
            filename, lineno, function_name, code = stack[-2]
            file=open(self.filename,"a")
            
            parameters=''
            res=dS.flexPatternSearch(code, '(', ')')
            code=code[res['indicesStart'][0]+1:
                      res['indicesStop'][0]]
            variables=code.split(',')
            
            for i in range(len(variables)):
                parameters+=variables[i]+'='+str(args[i])+'\t'
                
            
            text="\n"+"parameters,"+parameters+","+dt.datetime.now().strftime("%b-%d-%I:%M:%S%p")
            file.write(text)
            file.close()
            self.printlog(text)
            
            
# TODO
"""
store human readable date
Think about how to structure the file 
and how to store the information for easy
visualization of the process
have separate fields for activities, parameters, results
"""    
        
        
if __name__=="__main__":
    import numpy as np
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from sklearn.datasets.samples_generator import make_blobs
    
#     ins=instrumento(path=".",logname="meanShift-test.txt",printout=True)
    ins=instrumento()
    
    ins.act("Start")
    
    ###############################################################################
    # Generate sample data    
    ins.act("generating data")
    centers = [[1, 1], [-1, -1], [1, -1]]
    cluster_std=0.6
    n_samples=10000
    ins.params(centers,n_samples,cluster_std)
    X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)
    
    ###############################################################################
    # Compute clustering with MeanShift
    
    ins.act("Optimizing bandwith")
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    bin_seeding=True
    ins.act("Clustering")
    
    ins.params(bandwidth, bin_seeding)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    

    ins.sum(n_clusters_,"Number of estimated clusters")
    ###############################################################################
    # Plot result
    import matplotlib.pyplot as plt
    from itertools import cycle
    
    plt.figure(1)
    plt.clf()
    
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    ins.act("visualization")
    plt.show()
    ins.act("End")