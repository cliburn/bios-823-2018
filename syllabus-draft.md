# Syllabus for BIOS 823

BIOS 823 (Biomedical Big Data) explains what Big Data is, the types of Big Data often encountered in medical science (free text, time series, networks, genomic sequences, and images), and common strategies for the processing and analysis of Big Data. The core topics are **data** management (use of relational and NoSQL databases), **algorithms** (the mathematics and ideas behind common applications such as dimension reduction, clustering, collaborative filtering, classification and regression), **applications** (use of algorithms on text, time series, genomic, network and image data, including basic deep learning examples), and computational strategies to improve **performance** (code profiling, native code compilation, multi-core parallelism, distributed computing, and clout platforms).

## Learning objectives

After this course, the student will be able to

- [ ] Manage large data collections
  - [ ] Match data collections to appropriate SQL and NoSQL database types
  - [ ] Perform ETL operations to populate the database
  - [ ] Perform queries from the database
  - [ ] Convert data from one format to another
- [ ] Perform computationally intensive numerical operations
  - [ ] Solving $Ax = b$
  - [ ] Matrix factorization (solve linear systems, change of basis, projection and dimension reduction, low rank approximation and imputation)
  - [ ] Large-scale optimization with gradient descent algorithms
  - [ ] Apply numerical algorithms
- [ ] Decrease run-time by compiling to native code
  - [ ] Profile code to identify bottlenecks
  - [ ] Benchmark code for comparative evaluation
  - [ ] Apply JIT and AOT compilation
  - [ ] Wrap C++ code
- [ ] Use threads and processes for multi-core parallelization
  - [ ] Use parallelization in JIT and AOT code
  - [ ] Run parallel code interactively with `ipyparallel`
  - [ ] Using asynchronous programming to parallelize latency bound jobs
  - [ ] Using processes to parallelize compute bound jobs
- [ ] Perform big data computations on a distributed cluster
  - [ ] Understand the distributed computing ecosystem
  - [ ] Manipulate Spark RDDs and DataFrames
  - [ ] Use Spark MLLib for machine learning
  - [ ] Use Spark for streaming analysis
- [ ] Build and run deep learning pipelines
  - [ ] Explain concepts of deep learning
  - [ ] Build a CNN for image classification
  - [ ] Improve model performance
- [ ] Cross-cutting skills
  - [ ] Construct reproducible analysis pipelines
  - [ ] Use remote computing clusters (`slurm` on Duke Compute Cluster)
  - [ ] Use cloud computing platforms (AWS, GCE, Azure)

## S01 Data types, storage and retrieval

- Skills for data scientists
- Tools for data scientists
- Big data
  - Volume
  - Velocity
  - Variety
  - Veracity
- Types of data
  - Structured
  - Semi-structured
  - Unstructured
- Databases
  - Flat file
  - Hierarchical
  - Relational
  - Key-value
  - Document
  - Graph
  - Column family
- Example: Using `odo` to convert data types

## S02 Relational databases

- Relational database management systems (RDBMS)
  - Storage
  - Memory
  - Dictionary
  - Query language
- What is relational?
  - Main concepts
    - Tables
    - Columns
    - Rows
  - Secondary concepts
    - Constraints (identity and check)
    - Views
    - Indexes
- ACID transactions
  - Atomic
  - Consistent
  - Isolated
  - Durable
- The database schema
  - Well-behaved columns
    - Choosing column names
    - Fixing multi-part columns
    - Fixing multi-valued columns
    - Removing redundant columns
  - Well-behaved tables
    - Choosing table names
    - Existence of primary key
    - Check all columns belong to table

## S03 Using SQL

- Accessing databases
  - Using the CLI
  - Using a Python DBAPI driver
  - Using `sqlmagic`
- Using the data Dictionary
- Creating a database
- Data manipulation
  - Inserting new rows
  - Updating a row
  - Deleting a row
- Basic queries
  - Projection
  - Filtering
  - Cartesian product (joins)
  - Sorting
  - Aggregate functions
  - Group by
- Intermediate queries
  - Using explain
  - Using an index
  - Sub-queries
  - Window functions  

## S04 A tour of NoSQL databases

- Why NoSQL?
- BASE
  - Basically Available
  - Soft state
  - Eventual consistency
- Key-value
  - Python dictionaries
  - Example using `redis`
- Document
  - JSON documents
  - Example using `mongodb`
- Graph
  - Adjacency matrix, adjacency list  
  - Example using `neo4j`
- Column family
  - Columnar data stores `arrow`, `feather`, `parquet`
  - Example using `hbase`  
- Choosing between database types

## S05 Solving linear systems

- Why linear systems are important even if the world is nonlinear
- System of linear equations in matrix notation
- Solving $AX = B$
- Matrix factorization
- Geometric intuition and linear algebra
- Normal solution for linear least squares
- Example: Fitting a polynomial function to data

## S06 Dimension reduction

- Why dimension reduction?
- Manifolds
- PCA and factor analysis
- t-SNE and friends
- Example: Visualizing single cell data

## S07 Clustering and anomaly detection

- Why clustering?
- Hierarchical clustering and the distance matrix
- k-means
- Gaussian mixture Model
- Model selection for number of clusters
- Example: Anomaly detection by clustering

## S08 Recommender systems

- What is a recommender (collaborative filtering) system?
- SVD, linear algebra and the fundamental subspaces
- Alternating least squares (ALS)
- Example: MovieLens recommendations

## S09 Model fitting and optimization

- Review of calculus
- Univariate and multivariate optimization
- Gradient descent methods
- Newton and quasi-Newton methods
- Constrained optimization
- Example: Graph layout using a spring algorithm

## S10 Classification and regression

- Labeled data and supervised learning
- Under-fitting and over-fitting (Bias-variance trade-off)
- Regularization and generalization
- Cross-validation and out-of-sample prediction
- A pipeline for supervised learning
- Example: Using `sklearn` to classify MNIST digits

## S11 Analysis of text data

- Natural language processing (NLP)
- Bag of words
- One-hot and integer coding
- Tf-idf and document retrieval
- SVD and Latent Semantic Analysis
- Example Using `nltk` to extract named entities
- Example: Using `gensim` to classify newsgroups

## S12 Analysis of time series data

- Smoothing with exponential weighted averages
- Scan statistics
- Stationarity
- Trends and patterns
- Differencing
- Decomposition
- ACF and PACF
- Forecasting with ARIMA models
- Example: Using `prophet` for time series decomposition

## S13 Analysis of genomics data

- Simple DNA processing
- Motif finding
- Sequence alignment
- Example: Using `biopython` for genomic data bioinformatics

## S14 Analysis of image data

- Image formats
- Image to array
- Morphological operations
- Convolution and image filters
- Image segmentation
- Example: Usng `skimage` for color separation of histology images

## S15 Analysis of network data

- Graphs and networks
- Classic algorithms
  - Shortest path
  - Minimum spanning tree
  - Ford–Fulkerson algorithm
  - Graph laplacian
  - Example: Using `networkx` to find community structure

## S16 Deep learning primer

- A single unit
- Activation functions
- Layers and notation
- Forward propagation
- Back-propagation
- Automatic differentiation
- Cost function
- Stochastic and mini-batch gradient descent
- Regularization
- Deep Learning zoo (Dense, CNN, RNN, GAN)

## S17 Deep learning applications

- Classifying Fashion MNIST
- Inspecting features in layers
- Using `Auto-Keras`

## S18 JIT and AOT compilation

- Interpreted and compiled code
- Using `numba`
- Using `cython`
- Using `pybind11`

## S19 Multi-core parallelism

- Why mullit-core?
- Amdahl and Gustaffson laws
- Race conditions and deadlock
- Embarrassingly parallel problems
- Python's Global Interpreter Lock
- Using `concurrent.futures`
- Using `multiprocessing`

## S20 Asynchronous programming

- Why asynchronous?

## S21 Distributed computing and Hadoop/Spark

## S22 Data Frames and Spark SQL

## S23 Machine Learning with Spark MLLib

## S24 Spark Streaming

## S25 Cloud computing platforms