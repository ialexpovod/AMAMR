import numpy
from random import normalvariate

class Tikhonov(object):
	'''
	Solves Tikhonov regularization problem with the covariance of the weights as a prior.
	Q = inv(K.T * K + inv(L)*lambda) * K.T * R
	:parametr K: {array-like}, shape = [n_samples, n_features]
	:parametr R: array-like, shape = [n_samples] or [n_samples, n_targets]
	:param lambda: egularization parameter
	:param L: array-like, shape = [n_features, n_features] - covariance matrix of the prior
	:return: Q_hat: vector Q estimates
	'''	
	
	def __init__(self, K, R, L = None):
		'''
		Конструктор	
		'''
		self.K = numpy.array(K)
		self.R = numpy.array(R)
		self.L = numpy.array(L)
		
		
	def SVD(self, A, k = None, epsilon = 1e-10):
		'''
		Compute the singular value decomposition of a matrix A
		using the power method. A is the input matrix, and k
		is the number of singular values you wish to compute.
		If k is None, this computes the full-rank decomposition.
		'''
		
		self.A = A
		self.k = k
		self.epsilon = epsilon
	
		A = numpy.array(A)
		n, m = A.shape
		
		def randomUnitVector(n):
				
			'''
				The function return random list according by the norm.
				:parametr n: selecting from the minimum value of the number of rows and columnsю
				:return: list random vector dimensional n
			'''
			
			unnormalized = [normalvariate(mu = 0, sigma = 1) for _ in range(n)]
			norma = sum(item * item for item in unnormalized) ** 0.5 # numpy.linalg.norm(unNormalized)
			return [item / norma for item in unnormalized]
		
		def SVD1D(A, epsilon = 1e-10):
			
			A = numpy.array(A, dtype = float)
			n, m = A.shape
			x = randomUnitVector(min(n,m))
			lastV = None; currentV = x
			
			if n > m:
				B = numpy.dot(numpy.transpose(A), A)
			else:
				B = numpy.dot(A, numpy.transpose(A))
				
			iterations = 0
			while True:
				iterations += 1
				lastV = currentV
				currentV = numpy.dot(B, lastV)
				currentV = currentV / numpy.linalg.norm(currentV)
				
				if abs(numpy.dot(currentV, lastV)) > 1 - epsilon:
#					print("converged in {} iterations!".format(iterations))
					return currentV
				
		svdSoFar = []
		if k is None:
			k = min(n, m)
			
		for i in range(k):
			matrixFor1D = A.copy()
			
			for singularValue, u, v in svdSoFar[:i]:
				matrixFor1D = matrixFor1D - singularValue * numpy.outer(u, v)
				
			if n > m:
				v = SVD1D(matrixFor1D, epsilon = epsilon)  # next singular vector
				u_unnormalized = numpy.dot(A, v)
				sigma = numpy.linalg.norm(u_unnormalized)  # next singular value
				u = u_unnormalized / sigma
			else:
				u = SVD1D(matrixFor1D, epsilon=epsilon)  # next singular vector
				v_unnormalized = numpy.dot(A.T, u)
				sigma = numpy.linalg.norm(v_unnormalized)  # next singular value
				v = v_unnormalized / sigma
				
			svdSoFar.append((sigma, u, v))
			
		singularValues, us, vs = [numpy.array(x) for x in zip(*svdSoFar)]
		return singularValues, numpy.transpose(us), vs
	
	def Activity(self, L = None):
		
		
		KR = numpy.append(self.K, self.R[0:], axis=1)
	
		s, _, _ = self.SVD(KR)
		Lambda = s[1] ** 3
		if L is None:
#			L = numpy.eye(len(self.K[0]))
			L = numpy.array([[1, .5], [.5, 1]])
		KT = numpy.transpose(self.K)
		Q = numpy.dot(numpy.linalg.pinv(numpy.dot(KT, self.K) + L * Lambda), numpy.dot(KT, self.R))

		return Q


if __name__ == '__main__':	
	K = [[278.87e-3,82.17e-3],[0,41.79e-3]]
	R = [[3.022],[0.430]]
	d = Tikhonov(K, R)
	print(d.Activity())
