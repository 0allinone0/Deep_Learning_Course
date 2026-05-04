import numpy as np

x = np.array([
    [1, 2, 3, 0, 1],
    [4, 5, 6, 1, 2],
    [7, 8, 9, 2, 3],
    [1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4]
])

w = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

print("input shape:", x.shape)
print("kernel shape:", w.shape)

# Hyper-parameters
kernel_size = w.shape[0]
stride = 1
padding = 0

# Auto-calculation of output shape
input_size = x.shape[0]
output_size = (input_size + 2 * padding \
               - kernel_size) // stride + 1

out = np.zeros((output_size, output_size), dtype=int)

for i in range(output_size):
    for j in range(output_size):
        input_i = i * stride
        input_j = j * stride

        region = x[input_i:input_i+kernel_size, 
                   input_j:input_j+kernel_size]
        out[i, j] = np.sum(region * w)

        print(f"The center of window in terms of input coordinates: [{input_i}, {input_j}]")
        print(region)
        print("Weighted region:")
        print(region * w)
        print("The corresponding output (weighted sum):", out[i, j])
        print("-" * 30)

print("Final output")
print(out)
print("output shape:", out.shape)