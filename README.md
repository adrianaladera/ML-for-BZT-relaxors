# ML-for-BZT-relaxors
This repository accompanies the paper "*Machine learning reveals memory of the parent phases in ferroelectric relaxors* $Ba(Ti_{1-x},Zr_x)O_3$". It utilizes principal component analysis and $K$-means clustering to unveil subtle phases-- overlooked by the traditional thermodynamic approach-- of relaxor ferroelectrics in the family of solid solutions $Ba(Ti_{1-x},Zr_x)O_3$ (BZT).
  
  - dependencies: Python >= 3.0, Anaconda, Scikit Learn = 1.1.3
  
  All of the following scripts can be run in the root directory:
  
  `<PATH_TO_YOUR_DIR>/ML-for-BZT-relaxors/`
 
 ## $Ba(Ti_{1-x},Zr_x)O_3$ Data
 For each concentration x = Zr in BZT, we use Molecular Dynamics (MD) to anneal a BZT simulation supercell from 450 K to 10 K every 10 K steps. At each temperature we output a dipole pattern every 30,000 (MD) steps, which yields $P = 450$ dipole patterns overall per concentration. The dipole pattern is a $30 \times 30 \times 30$ supercell of BZT with $27,000$ dipoles, and each dipole has information for its 3 position coordinates and 3 vector components. Together, this yields 6 variables for each dipole and $N=6 \times$ $27,000$ total variables for any given supercell. The input data for each concentration x of BZT is therefore a 2D array with dimensions $P \times N$, where each row corresponds to a single supercell.
 
 Data for all concentrations of x, including concentrations in the Supplementary Materials, can be obtained [here](https://drive.google.com/drive/folders/1sL33n8ptJidefb3jYnKVe-lRwkRETX-a?usp=sharing). Ensure that the downloaded data folder, titled `data_tables`, is inside the root directory (i.e. the same level as `kmeans-BZT.py`. 
 
  ## Principal component analysis
  Principal component analysis accepts data of size $P \times N$ along with the number of desired principal components (which can be received as a percentage or an integer number) and reduces the original data to a size of $P \times M$, where $M$ is the number of principal components. The results in the paper recognize the $M$ that yield a 99% variance (i.e. 99% of the data can be explained within the first $M$ principal components).
 
## $K$-means clustering
 `kmeans-BZT.py` is an unsupervised learning program that utilizes the $K$-means clustering algorithm, an unsupervised algorithm that groups data into cluster groups without human input other than a $K$-value, which dictates how many clusters the algorithm should group the data points into. It accepts data as an $P \times N$ array, where $P$ is the number of samples and $N$ is the dimension, i.e. variables of each sample. In this case, $N=M$, where we input the PCA-reduced data set of size $P \times M$ into the $K$-means clustering algorithm.
 
 ## Output
 When running the workflow, a new directory for each concentration is created if it does not already exist. Each folder will produce two files, `k_value_selection.csv` and `kmeans_results.csv`. Below is an explanation for each of the key-value pairs listed under each resulting csv file.
 
 ### k_value_selection.csv
 - `k`: the $K$-value for a specific run of the $K$-means clustering algorithm
  - `distortion`: the average distance of each point to their respective centroid
   - `inertia`: the sum of the squared distances of each point to their respective centroid
 
 ### kmeans_results.csv
  - `temperature`: temperature in Kelvin of a supercell (row)
   - `cf_pattern`: for a dipole pattern outputed every 30,000 out of 300,000 MD steps, the integer number indicating which output is given for a supercell
   - `PCA-x`: the first principal component out of $M$ principal components
   - `PCA-y`: the second principal component out of $M$ principal components
- `k$a$,pred`: if $\alpha$ is a $K$-value, then an integer label corresponding with potential values [0, $a$] to the cluster prediction for a supercell is given
- `k$a$, x`: the "coordinate" in terms of the first principal component of the supercell with respect to its nearest centroid
- `k$a$, y`: the "coordinate" in terms of the second principal component of the supercell with respect to its nearest centroid
- `k$a$, centx`: the centroid "coordinate" in terms of the first principal component 
- `k$a$, centy`: the centroid "coordinate" in terms of the second principal component 
 
 
 ## Disclaimer
 All machine learning code for this paper was written using [Scikit-learn](https://scikit-learn.org/stable/). Learn more about [principal component analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) and [$K$-means clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) here.

  

