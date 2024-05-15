
# from sklearn.utils.multiclass import check_classification_targets
# from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.exceptions import NotFittedError
# from mkl.validation import check_KL_Y
# from mkl.exceptions import BinaryProblemError
# from mkl.multiclass import OneVsOneMKLClassifier, OneVsRestMKLClassifier
import torch
from sklearn.svm import SVC

# from mkl.model import Solution, Cache, TwoStepMKL
from mkl.evaluate import frobenius, margin
import numpy as np
import torch
from mkl.util import summation




class Solution():
	def __init__(self, weights, objective, ker_matrix, dual_coef, bias, **kwargs):
		self.Y = None
		self.weights 	= weights
		self.objective 	= objective
		self.ker_matrix = ker_matrix
		self.dual_coef 	= dual_coef
		self.bias		= bias
		self.__dict__.update(kwargs)


class Model():

	func_form  = None   # a function which takes a list of kernels and their weights and returns the combination
	n_kernels  = None   # the number of kernels used in combination
	KL 		   = None 	# the kernels list
	solution   = None 	# solution of the algorithm

	def __init__(self):
		self.classes_ = None
		self.func_form = summation
		self.theta = 0.0
		self.min_margin = 1e-4
		self.learning_rate = 0.01
		self.max_iter = 1000
		self.learner = SVC(C=0.5)

	# def fit(self, KL, Y):


	def fit(self, KL, Y):
		self.KL = KL
		self.Y = Y
		self.n_kernels = len(self.KL)
		self.classes_ = self.Y.unique()
		self.solution = self.fit_model()  # call combine_kernels without re-preprocess
		self.learner.fit(self.solution.ker_matrix, self.Y)
		return self


	def predict(self, KL):
		return self.learner.predict(self.func_form(KL,self.solution.weights))

	def initialize_optimization(self):
		Q = torch.tensor([[self.KL[j].flatten() @ self.KL[i].flatten() for j in range(self.n_kernels)] for i in
						  range(self.n_kernels)])
		Q /= (torch.diag(Q).sum() / self.n_kernels)

		self.Q = Q
		self.Y = torch.tensor([1 if y == self.classes_[1] else -1 for y in self.Y])


		weights = torch.ones(self.n_kernels, dtype=torch.double)/self.n_kernels
		ker_matrix = self.func_form(self.KL, weights)
		mar, gamma = margin(
			ker_matrix, self.Y,
			return_coefs=True,
			# solver=self._solver,
			# max_iter        = self.max_iter*10,
			# tol=self.tolerance
			)

		yg = gamma.T * self.Y

		self.gamma = gamma
		self.margin = mar
		bias = 0.5 * (gamma @ ker_matrix @ yg).item()

		print(weights.size(), yg.size(), weights.T.size(), yg.T.size(), ker_matrix.size(), Q.size(), len(self.KL))

		obj = (yg.T @ ker_matrix @ yg).item() + (weights @ Q @ weights).item() * self.theta * .5
		print('ok')
		return Solution(
			weights=weights,
			objective=obj,
			ker_matrix=ker_matrix,
			bias = bias,
			dual_coef= gamma
		)

	def fit_model(self):
		# self.learning_rate = self.initial_lr
		self.solution = self.initialize_optimization()
		# self.convergence = False
		# multiplier = -1

		step = 0
		while step < self.max_iter:
			step += 1
			current_solution = self.perform_step()
			self.solution = current_solution
		return self.solution

	def perform_step(self):
		Y = self.Y

		# positive margin constraint
		if self.margin <= self.min_margin: # prevents initial negative margin. Looking for a better solution
			return self.solution

		# weights update
		yg = self.gamma.T * Y
		grad = torch.tensor([self.theta * (qv @ self.solution.weights).item() + (yg.T @ K @ yg).item() \
							 for qv, K in zip(self.Q, self.KL)])
		beta = self.solution.weights  # .log()
		beta = beta + self.learning_rate * grad
		beta_e = beta  # .exp()

		weights = beta_e
		weights[weights < 0] = 0
		weights /= sum(beta_e)

		# compute combined kernel
		ker_matrix = self.func_form(self.KL, weights)

		# margin (and gamma) update
		mar, gamma = margin(
			ker_matrix, Y,
			return_coefs=True,
			# solver=self._solver,
			# max_iter        = self.max_iter,
			# tol=self.tolerance
		)

		# positive margin constraint
		if mar <= self.min_margin:
			return self.solution

		# compute objective and bias
		yg = gamma.T * Y

		obj = (yg.T @ ker_matrix @ yg).item() + self.theta * .5 * (
				weights.view(1, len(weights)) @ self.Q @ weights).item()
		bias = 0.5 * (gamma @ ker_matrix @ yg).item()

		# update cache
		self.gamma = gamma
		self.margin = mar

		return Solution(
			weights=weights,
			objective=obj,
			ker_matrix=ker_matrix,
			dual_coef=gamma,
			bias=bias,
		)


