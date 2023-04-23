import numpy as np

x_i = np.array([[1,2],[2,2],[4,5]])
r_i = np.array([[0],[1],[0]])

w_i = 2
print(w_i)
h_number = 2
print(h_number)
h = np.array([[h_number,h_number]])

Multiply_x_i_w_i = w_i*x_i
list_test = []
print(Multiply_x_i_w_i)
print(h)


times = 10
for i in Multiply_x_i_w_i:
    if Multiply_x_i_w_i.any()>h.any():
        list_test.append(1)
    else:
        list_test.append(0)


print(list_test)
