<p>In both project implementations I wrote a class that contained a K-Means Algorithm, data visualization, and centroid coordinate table visualization methods among some other helper methods. </p>
<p>Strategy 1 used random k data points as it's centroid, then runs the K-Means Algorithm on the data and centroids</p>
<p>Strategy 2 implementation used a random initial data point as a centroids, then calculates the remaining k-1 centroids by maximizing distance to all remaining data points, then runs the K-Means Algorithm on the data and centroids</p>
<h2>K-Means Output</h2>
<p> This is an example of a K-Means output with a k value = 10</p>
<p>The plot shows the centroids (larger circles with a border) and clustered data points</p>
<p>The centroids are the locations of the center of each cluster</p>
<br></br>
<img src="/Project2/Examples/k10.png">
<br></br>
<h2>K-Means Output Plot & Centroids </h2>
<p>This is a plot of the centroids of this data set along with the centroid for each cluster (and the values) k = 6</p>
<img src="/Project2/Examples/k6.png">
<br></br>
<img src="/Project2/Examples/k6Centroids.PNG">
