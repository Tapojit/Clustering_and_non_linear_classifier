import numpy as np
import matplotlib.pyplot as plt
import cluster_utils
import cluster_class
from sklearn.cluster import KMeans

#Load train and test data
train = np.load("../../Data/ECG/train.npy")
test = np.load("../../Data/ECG/test.npy")

#Create train and test arrays
Xtr = train[:,0:-1]
Xte = test[:,0:-1]
Ytr = np.array(map(int, train[:,-1]))
Yte = np.array( map(int, test[:,-1]))

#Calculating cluster quality for 1-40 clusters using k-means clustering
clustering_quality=[]
for i in range(1, 41):
    kmeans=KMeans(n_clusters=i, random_state=0).fit(Xtr)
    Z=kmeans.labels_
    clustering_quality.append(cluster_utils.cluster_quality(Xtr, Z, i))


#Using optimal cluster number of 35 to generate waveforms for 35 clusters
kmeans=KMeans(n_clusters=35, random_state=0).fit(Xtr)
Z=kmeans.labels_
prop=cluster_utils.cluster_proportions(Z, 35)
means=cluster_utils.cluster_means(Xtr, Z, 35)
cluster_utils.show_means(means, prop).savefig("../Figures/waveform.png")

#Calculating prediction error for 1-40 clusters
scores=[]
for i in range(1, 41):
    v=cluster_class.cluster_class(i)
    mod=v.fit(Xtr,Ytr)
    predict=v.predict(Xte)
    scores.append(v.score(Xte,Yte))








def bar_plot(x_lab, y_lab, x, y, title):
    plt.figure(1, figsize=(10,4))  #6x4 is the aspect ratio for the plot
    plt.bar(x, y, align='center') #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel(y_lab) #Y-axis label
    plt.xlabel(x_lab) #X-axis label
    plt.title(title) #Plot title
    plt.xlim(0,36) #set x axis range
    plt.ylim(0,0.2) #Set yaxis range
    
    #Set the bar labels
    plt.gca().set_xticks(x) #label locations
    
    #Make sure labels and titles are inside plot area
    plt.tight_layout()
    
    #Save the chart
    plt.savefig("../Figures/"+title+".png")
    print "barplot image generated"
# bar_plot("Cluster no.", "Cluster proportions ", range(1,36), prop, "Cluster proportions VS. Cluster no")


def line_plot(x_lab, y_lab, x, y, title):
    values =y
    inds   =x
    #Plot a line graph
    plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.plot(inds,values,'or-', linewidth=3) #Plot the first series in red with circle marker
    
    #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel(y_lab) #Y-axis label
    plt.xlabel(x_lab) #X-axis label
    plt.title(title) #Plot title
    plt.xlim(0, 40) #set x axis range
    plt.ylim(0,1) #Set yaxis range
    
    #Make sure labels and titles are inside plot area
    plt.tight_layout()
    
    #Save the chart
    plt.savefig("../Figures/"+title+".png")
    
    print "Line graph image generated"

# line_plot("No. of clusters", "Cluster quality value", range(1,41), clustering_quality, "Clustering quality value VS. No. of clusters")

# line_plot("Prediction error", "Cluster no.", range(1,41), scores, "Prediction error VS. Cluster no")




