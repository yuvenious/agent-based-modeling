import time

import numpy as np
import pandas as pd

import recycle_module_py
import recycle_module_cy
import recycle_module_cy_default

import matplotlib.pyplot as plt

kwargs = {"n_mun": 10}
n = 100

pys = []
cys = []
cys_d = []

print("=============Python==============")
for i in range(n):
	# start = time.time()
	model = recycle_module_py.Model(**kwargs)
	# end = time.time()
	# print("init_time:", end-start)
	# start = time.time()
	model.setup()
	# end = time.time()
	# print("setup_time:", end-start)
	start = time.time()
	for i in range(240):
	    model.update()
	end = time.time()
	py = end-start
	# print("run_time:", py)
	pys.append(py)
print("=============Cython==============")
for i in range(n):
	# start = time.time()
	model = recycle_module_cy.Model(**kwargs)
	# end = time.time()
	# print("init_time:", end-start)
	# start = time.time()
	model.setup()
	# end = time.time()
	# print("setup_time:", end-start)
	start = time.time()
	for i in range(240):
	    model.update()
	end = time.time()
	cy = end-start
	# print("run_time:", cy)
	cys.append(cy)
print("=============Cython_Default==============")
for i in range(n):
	# start = time.time()
	model = recycle_module_cy_default.Model(**kwargs)
	# end = time.time()
	# print("init_time:", end-start)
	# start = time.time()
	model.setup()
	# end = time.time()
	# print("setup_time:", end-start)
	start = time.time()
	for i in range(240):
	    model.update()
	end = time.time()
	cy = end-start
	# print("run_time:", cy)
	cys_d.append(cy)


data = [pys, cys, cys_d]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(data)
ax.set_xticklabels(["python", "cython", "cython_default"])
plt.show()
