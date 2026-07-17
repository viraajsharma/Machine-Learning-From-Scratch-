Linear Algebra 

Why Linear Algebra?
Linear algebra is the mathematics of data and is a universal prerequisite for a deep understanding of machine learning
. Almost every algorithm operates on vectors and matrices to perform efficient computations and optimizations
.
Examples:
Linear & Logistic Regression: Models relationships using matrix forms to solve for optimal parameters
.
PCA (Principal Component Analysis): Uses matrix decompositions to reduce the dimensionality of high-dimensional data
.
Neural Networks: Heavily depend on matrix multiplications, weight initialisation, and gradient descent to train deep models
.
Scalars
Definition: A scalar is a single numerical value that has magnitude but no direction
. In mathematical notation, they are typically represented by lowercase italic letters
.
Example: A single number such as k=3
.
Applications: In machine learning, scalars are used to adjust weights in a model or define the learning rate during training
.
Vectors
Definition: Vectors are special objects that can be added together and multiplied by scalars
. They represent quantities that have both magnitude and direction, often visualized as arrows in space
.
Notation: Vectors are commonly denoted by bold lowercase letters like x or v
. By default, they are usually represented as column vectors (vertical arrays)
.
Magnitude: Also called the norm (∥v∥), it represents the length of the vector
. For a vector in R 
n
 , it is calculated as the square root of the sum of the squares of its components
.
Unit Vector: A vector with a magnitude of exactly 1
. Any non-zero vector can be converted into a unit vector by dividing it by its magnitude
.
Dot Product: A measure of the similarity of directions between two vectors
. It is calculated by multiplying matching elements and summing the results
.
Angle Between Vectors: The dot product reveals the angle θ through the relationship v⋅w=∥v∥∥w∥cosθ
. If the dot product is zero, the vectors are perpendicular (orthogonal)
.
Applications in ML: Data points are represented as feature vectors, where each component is a specific attribute of the data
.
Matrices
Definition: A matrix is a rectangular array of numbers arranged in rows and columns
.
Dimensions: A matrix with m rows and n columns is called an m×n matrix
.
Row Vector & Column Vector: A 1×n matrix is a row vector, while an m×1 matrix is a column vector
.
Identity Matrix (I): A square matrix with 1s on the main diagonal and 0s everywhere else
. Multiplying any matrix by I leaves it unchanged
.
Transpose (A 
⊤
 ): Created by flipping a matrix across its diagonal, turning rows into columns and vice versa
.
Inverse (A 
−1
 ): A square matrix that, when multiplied by the original matrix, produces the identity matrix (AA 
−1
 =I)
. It exists only if the matrix is "non-singular" (determinant 

=0)
.
Rank: The number of linearly independent rows or columns in a matrix
.
Applications: Matrices represent datasets (where rows are samples and columns are features) and linear transformations
.
Matrix Operations
Addition & Subtraction: These operations are performed element-wise, meaning you add or subtract corresponding entries in matrices of the same dimensions
.
Multiplication (Scalar): Multiplying a matrix by a scalar scales every entry in the matrix by that number
.
Element-wise Multiplication: Often called "array multiplication" in programming, it involves multiplying corresponding elements but does not follow the standard rules of matrix algebra
.
Broadcasting: This involves performing element-wise operations on arrays of different sizes by expanding the smaller array to match the larger one (e.g., adding a scalar to every element of a vector)
.
Matrix Multiplication
Rules: Two matrices can only be multiplied if the number of columns in the first equals the number of rows in the second
. The result of (m×n)×(n×p) is an m×p matrix
.
Example: To find an entry in the resulting matrix, you take the dot product of the corresponding row from the first matrix and the column from the second
.
Complexity: For two n×n square matrices, multiplication requires roughly n 
3
  operations
.
Applications: Used for feature transformations, neural network layer operations, and solving systems of linear equations
.
Eigenvalues
Definition: An eigenvalue (λ) is a scalar that represents how much a transformation stretches or compresses space along a specific direction
.
Why they matter: They characterize the fundamental properties of a linear transformation
. A matrix is invertible only if all its eigenvalues are non-zero
.
PCA: In dimensionality reduction, PCA identifies the directions (eigenvectors) with the largest eigenvalues, as these capture the most variance in the data
.
Covariance Matrix: A symmetric matrix whose eigenvalues represent the variance explained by each principal component
.
Eigenvectors
Definition: An eigenvector is a non-zero vector that does not change direction when a linear transformation is applied to it; it is only scaled by its corresponding eigenvalue
.
Intuition: They represent the "principal axes" or the most stable directions of a transformation
.
Applications: Used in facial recognition (eigenfaces), Google's PageRank algorithm, and vibration analysis
.
Singular Value Decomposition (SVD)
Overview: SVD is the "fundamental theorem of linear algebra" because it can be applied to any matrix, including non-square ones
. It factors a matrix A into UΣV 
⊤
 , where U and V are orthogonal matrices and Σ contains singular values
.
Applications:
Data Compression: Approximates large matrices using fewer components (low-rank approximation)
.
Noise Filtering: Removes small singular values that often represent noise in a dataset
.
Latent Semantic Analysis: Identifies hidden structures, such as movie themes in user rating data
.
Summary
To understand Deep Learning, you must progress from the simplest building blocks to complex decompositions:
Scalars are individual numbers used for scaling
.
Vectors represent data points in space with magnitude and direction
.
Matrices act as containers for datasets and represent the linear rules (transformations) applied to those vectors
.
Matrix Multiplication is the core engine that powers neural network layers
.
Eigen-analysis and SVD allow us to break down complex matrices into their most important parts, enabling data compression and dimensionality reduction
.
By mastering these concepts, you gain the ability to manipulate high-dimensional data and optimize the complex functions required for artificial intelligence
.