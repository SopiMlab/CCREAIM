# PARAMETERS
down_t = 7
stride_t = 2
block_dilation_growth_rate = 3
block_depth = 4
block_kernel_size = 3

j_i = 1
r_out = 1

for i_enc_block in range(down_t):
    kernel_size = stride_t * 2
    r_out = r_out + (kernel_size - 1) * j_i
    j_i = j_i * stride_t
    for i_res_block in range(block_depth):
        d = block_dilation_growth_rate**i_res_block
        r_out = r_out + (block_kernel_size - 1) * d * j_i

print(f"Total receptive field {r_out}")
