import os
import sys
import numpy             as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

	resdir = sys.argv[1]
	maxidx = int(sys.argv[2])

	results = []

	for idx in range(maxidx + 1):
		name = '%s/%02i'%(resdir, idx)

		with open(name, 'r') as file:
			text = file.read()

		lines = text.strip().split('\n')

		current_results = []
		for line in lines:
			ops, ns = line.split(' ')
			current_results.append((int(ops), int(ns)))

		results.append(current_results)

	ops      = []
	minimums = []
	maximums = []
	means    = []
	stds     = []
	for opcount in range(len(results[0])):
		ops.append(results[0][opcount][0])
		times = []
		for run in results:
			times.append(run[opcount][1])

		times = np.array(times)

		minimums.append(times.min())
		maximums.append(times.max())
		means.append(times.mean())
		stds.append(times.std())

	fig, ax = plt.subplots(1, 2)
	

	# ax[0].plot(ops, minimums)
	# ax[0].set_yscale('log')
	# ax[0].set_xscale('log')
	# ax[0].set_title('Minimum')

	# ax[1].plot(ops, maximums)
	# ax[1].set_yscale('log')
	# ax[1].set_xscale('log')
	# ax[1].set_title('Maximum')

	# ax[2].plot(ops, means)
	# ax[2].set_yscale('log')
	# ax[2].set_xscale('log')
	# ax[2].set_title('Mean')

	_mean, = ax[0].plot(ops, means)
	_min,  = ax[0].plot(ops, minimums)
	_max,  = ax[0].plot(ops, maximums)
	ax[0].set_yscale('log')
	ax[0].set_xscale('log')
	ax[0].set_title('Time')
	ax[0].set_xlabel('Floating Point Operations')
	ax[0].set_ylabel('Time [ns]')
	ax[0].grid(which='major')
	ax[0].grid(which='minor', alpha=0.3)

	ax[0].legend([_mean, _min, _max], ["Mean", "Minimum", "Maximum"])

	flops    = np.array(ops) / (np.array(means) / 1e9)
	maxflops = np.array(ops) / (np.array(minimums) / 1e9)
	minflops = np.array(ops) / (np.array(maximums) / 1e9)

	_mean, = ax[1].plot(ops, flops)
	_min,  = ax[1].plot(ops, minflops)
	_max,  = ax[1].plot(ops, maxflops)
	#ax[0].set_yscale('log')
	#ax[0].set_xscale('log')
	ax[1].set_title('FLOPS')
	ax[1].set_xlabel('Floating Point Operations')
	ax[1].set_ylabel('FLOPS')
	ax[1].grid(which='major')
	ax[1].grid(which='minor', alpha=0.3)
	ax[1].legend([_mean, _min, _max], ["Mean", "Minimum", "Maximum"])

	fig.tight_layout()

	print("Maximum FLOPS per core:      %f GFlops"%(flops.max() / 1e9))
	print("Best Run Max FLOPS per core: %f GFlops"%(maxflops.max() / 1e9))

	plt.show()