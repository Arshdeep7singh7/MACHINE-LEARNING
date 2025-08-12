import numpy as np

arr = np.array([1, 2, 3, 6, 4, 5])
reversed_arr = arr[::-1]
print("Reversed array:", reversed_arr)
array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])

# Method 1: using flatten()
flat1 = array1.flatten()

# Method 2: using ravel()
flat2 = array1.ravel()

print("Flattened with flatten():", flat1)
print("Flattened with ravel():", flat2)
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])

# Element-wise comparison
comparison = np.array_equal(arr1, arr2)
print("Are arr1 and arr2 equal?:", comparison)
# i.
x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1])
(unique, counts) = np.unique(x, return_counts=True)
max_count_index = np.argmax(counts)
most_freq_x = unique[max_count_index]
indices_x = np.where(x == most_freq_x)[0]
print("Most frequent value in x:", most_freq_x)
print("Indices of most frequent value in x:", indices_x)

# ii.
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])
(unique_y, counts_y) = np.unique(y, return_counts=True)
max_count_index_y = np.argmax(counts_y)
most_freq_y = unique_y[max_count_index_y]
indices_y = np.where(y == most_freq_y)[0]
print("Most frequent value in y:", most_freq_y)
print("Indices of most frequent value in y:", indices_y)
gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]')

# i. Sum of all elements
total_sum = gfg.sum()

# ii. Sum row-wise
row_sum = gfg.sum(axis=1)

# iii. Sum column-wise
col_sum = gfg.sum(axis=0)

print("Total sum:", total_sum)
print("Row-wise sum:\n", row_sum)
print("Column-wise sum:\n", col_sum)
n_array = np.array([[55, 25, 15], [30, 44, 2], [11, 45, 77]])

# i. Sum of diagonal elements
diag_sum = np.trace(n_array)

# ii. Eigenvalues
eig_vals = np.linalg.eigvals(n_array)

# iii. Eigenvectors
eig_vecs = np.linalg.eig(n_array)[1]

# iv. Inverse of matrix
inv_matrix = np.linalg.inv(n_array)

# v. Determinant of matrix
det_matrix = np.linalg.det(n_array)

print("Diagonal sum:", diag_sum)
print("Eigenvalues:", eig_vals)
print("Eigenvectors:\n", eig_vecs)
print("Inverse matrix:\n", inv_matrix)
print("Determinant:", det_matrix)
# i.
p1 = np.array([[1, 2], [2, 3]])
q1 = np.array([[4, 5], [6, 7]])

product1 = np.dot(p1, q1)
covariance1 = np.cov(p1.flatten(), q1.flatten())

print("Product of p1 and q1:\n", product1)
print("Covariance between p1 and q1:\n", covariance1)

# ii.
p2 = np.array([[1, 2], [2, 3], [4, 5]])
q2 = np.array([[4, 5, 1], [6, 7, 2]])

product2 = np.dot(p2, q2)
covariance2 = np.cov(p2.flatten(), q2.flatten())

print("Product of p2 and q2:\n", product2)
print("Covariance between p2 and q2:\n", covariance2)
x = np.array([[2, 3, 4], [3, 2, 9]])
y = np.array([[1, 5, 0], [5, 10, 3]])

# inner product between corresponding rows
inner_product = np.inner(x, y)

# outer product of flattened arrays
outer_product = np.outer(x.flatten(), y.flatten())

# Cartesian product (all combinations) using meshgrid and stacking
cartesian_product = np.array(np.meshgrid(x.flatten(), y.flatten())).T.reshape(-1, 2)

print("Inner product:\n", inner_product)
print("Outer product:\n", outer_product)
print("Cartesian product shape:", cartesian_product.shape)
print("Cartesian product sample:\n", cartesian_product[:10])
array = np.array([[1, -2, 3], [-4, 5, -6]])

# i. Element-wise absolute value
abs_array = np.abs(array)

# ii. Percentiles
percentiles_flat = np.percentile(array.flatten(), [25, 50, 75])
percentiles_col = np.percentile(array, [25, 50, 75], axis=0)
percentiles_row = np.percentile(array, [25, 50, 75], axis=1)

print("Absolute array:\n", abs_array)
print("Percentiles (flattened):", percentiles_flat)
print("Percentiles (columns):\n", percentiles_col)
print("Percentiles (rows):\n", percentiles_row)
mean_flat = np.mean(array.flatten())
median_flat = np.median(array.flatten())
std_flat = np.std(array.flatten())

mean_col = np.mean(array, axis=0)
median_col = np.median(array, axis=0)
std_col = np.std(array, axis=0)

mean_row = np.mean(array, axis=1)
median_row = np.median(array, axis=1)
std_row = np.std(array, axis=1)

print("Mean (flattened):", mean_flat)
print("Median (flattened):", median_flat)
print("Std (flattened):", std_flat)

print("Mean (columns):", mean_col)
print("Median (columns):", median_col)
print("Std (columns):", std_col)

print("Mean (rows):", mean_row)
print("Median (rows):", median_row)
print("Std (rows):", std_row)
a = np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])

floor_vals = np.floor(a)
ceil_vals = np.ceil(a)
trunc_vals = np.trunc(a)
round_vals = np.round(a)

print("Floor values:", floor_vals)
print("Ceiling values:", ceil_vals)
print("Truncated values:", trunc_vals)
print("Rounded values:", round_vals)
array = np.array([10, 52, 62, 16, 16, 54, 453])

# i. Sorted array
sorted_array = np.sort(array)

# ii. Indices of sorted array
sorted_indices = np.argsort(array)

# iii. 4 smallest elements
smallest_4 = np.partition(array, 3)[:4]

# iv. 5 largest elements
largest_5 = np.partition(array, -5)[-5:]

print("Sorted array:", sorted_array)
print("Indices of sorted array:", sorted_indices)
print("4 smallest elements:", smallest_4)
print("5 largest elements:", largest_5)
array2 = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])

# i. Integer elements only (check if value is integer)
int_elements = array2[array2 == array2.astype(int)]

# ii. Float elements only (not integers)
float_elements = array2[array2 != array2.astype(int)]

print("Integer elements only:", int_elements)
print("Float elements only:", float_elements)
from PIL import Image
import numpy as np

def img_to_array(path):
    img = Image.open(path)
    arr = np.array(img)
    
    if len(arr.shape) == 2:
        # Grayscale
        np.savetxt('grayscale_image.txt', arr, fmt='%d')
        print("Saved grayscale image array to grayscale_image.txt")
    elif len(arr.shape) == 3 and arr.shape[2] == 3:
        # RGB
        # Save as CSV with 3 channels separated by commas
        np.savetxt('rgb_image.txt', arr.reshape(-1, 3), fmt='%d', delimiter=',')
        print("Saved RGB image array to rgb_image.txt")
    else:
        print("Image format not supported")

# Example usage:
# img_to_array('path_to_image.jpg')
# For grayscale image
grayscale_array = np.loadtxt('grayscale_image.txt', dtype=int)
print("Loaded grayscale array shape:", grayscale_array.shape)

# For RGB image
rgb_array = np.loadtxt('rgb_image.txt', delimiter=',', dtype=int)
# Reshape back to original image dimensions if known (example 100x100 pixels):
# rgb_array = rgb_array.reshape((100, 100, 3))
print("Loaded RGB array shape:", rgb_array.shape)
