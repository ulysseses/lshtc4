import numpy as np

import similarity
import preproc

similarity.transform_tfidf(train_X)

k = 5 # or something else...


def optimized_ranks(scores, pscores, label_counter, w1, w2, w3, w4):
	''' w1..w4 are weights corresponding to x1..x4 '''
	ranks_dict = {}
	for c in scores:
		x1 = np.log(max(scores[c]))
		x2 = np.log(sum(pscores[c]))
		x3 = np.log(sum(scores[c]))
		x4 = np.log(len(scores[c])/len(label_counter[c]))
		ranks_dict[c] = w1*x1 + w2*x2 + w3*x3 + w4*x4
	return ranks_dict