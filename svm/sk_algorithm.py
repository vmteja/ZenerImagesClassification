from math import sqrt

import pickle

from svm import convex_hull


class UnTrainableException(Exception):
    pass

class SKAlg():
    def __init__(self, kernal):
        self.kernal = kernal
        self.alpha = []
        self.m_positive = []
        self.m_negative = []
        self.lmbda = 1

    def train(self, X, I_positive, I_negative, epsilon, max_updates, lmbda = 1):
        # STEP 1: Initialization
        self.lmbda = lmbda
        if len(X) == 0:
            raise UnTrainableException("Too less training inputs")

        if len(I_positive) == 0 or len(I_negative) == 0:
            raise UnTrainableException("Too less positive/negative labels")

        if len(X) > (len(I_positive) + len(I_negative)):
            raise UnTrainableException("Not all training inputs are labeled")

        X_dash, self.m_positive, self.m_negative = convex_hull.scale(X, I_positive, I_negative, lmbda)
        self.alpha = [0 for i in xrange(len(X_dash))]
        self.alpha[I_positive[0]] = 1
        self.alpha[I_negative[0]] = 1
        K = self.kernal

        x_dash_i1 = X_dash[I_positive[0]]
        x_dash_j1 = X_dash[I_negative[0]]

        self.A = K(x_dash_i1, x_dash_i1)
        self.B = K(x_dash_j1, x_dash_j1)
        C = K(x_dash_i1, x_dash_j1)
        D = [K(X_dash[i], x_dash_i1) for i in xrange(len(X_dash))]
        E = [K(X_dash[i], x_dash_j1) for i in xrange(len(X_dash))]

        n_updates = 0
        while True:
            # STEP 2: Stop Condition
            sqrt_a_b_2c = sqrt(self.A+self.B-2*C)
            m = lambda i: (D[i]-E[i]+self.B-C)/sqrt_a_b_2c if i in I_positive else (E[i]-D[i]+self.A-C)/sqrt_a_b_2c

            t = 0
            mt = m(0)
            for i in xrange(1, len(X_dash)):
                mi = m(i)
                if mi < mt:
                    t = i
                    mt = mi
            print n_updates, sqrt_a_b_2c
            if (sqrt_a_b_2c-mt) < epsilon or n_updates == max_updates:
                break

            # STEP 3: Adaptation
            if t in I_positive:
                q = min(1, (self.A-D[t]+E[t]-C)/(self.A+K(X_dash[t],X_dash[t])-2*(D[t]-E[t])))
                for i in xrange(len(I_positive)):
                    doe = 1 if i == t else 0
                    self.alpha[i] = (1-q) * self.alpha[i] + q * doe
                self.A = self.A * (1-q)**2 + 2 * (1-q) * q * D[t] + (q**2) * K(X_dash[t], X_dash[t])
                C = (1-q) * C + q * E[t]
                D = [(1-q) * D[i] + q * K(X_dash[i], X_dash[t]) for i in xrange(len(X_dash))]
            else:
                q = min(1, (self.B - E[t] + D[t] - C) / (self.B + K(X_dash[t], X_dash[t]) - 2 * (E[t] - D[t])))
                for i in xrange(len(I_negative)):
                    doe = 1 if i == t else 0
                    self.alpha[i] = (1 - q) * self.alpha[i] + q * doe
                self.B = self.B * (1 - q) ** 2 + 2 * (1 - q) * q * E[t] + (q ** 2) * K(X_dash[t], X_dash[t])
                C = (1 - q) * C + q * D[t]
                E = [(1 - q) * E[i] + q * K(X_dash[i], X_dash[t]) for i in xrange(len(X_dash))]
            n_updates += 1

    def serialize(self, output_file_name):
        svm_model = {'m_positive': self.m_positive, 'm_negative': self.m_negative, 'lamda': self.lmbda,
                     'A': self.A, 'B': self.B, 'alphas': self.alpha}
        with open(output_file_name, "wb") as f:
            pickle.dump(svm_model, f)
        f.close()