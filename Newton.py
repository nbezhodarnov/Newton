import numpy as np
import matplotlib.pyplot as plt
import math
	
class Newton():
	x = np.array([], dtype = float)
	y = np.array([], dtype = float)
	n = 0
	
	def __init__(self, x_array_input, y_array_input, count):
		self.x = x_array_input
		self.y = y_array_input
		self.n = count
		
	def __finite_difference(self, i, k):
		if (k == 0):
			return self.y[i]
		return self.__finite_difference(i + 1, k - 1) - self.__finite_difference(i, k - 1)
		
	def __forward_interpolation(self, x_input):
		t = (x_input - self.x[0]) / (self.x[1] - self.x[0])
		multiplier = 1
		result = 0
		for i in range(self.n):
			result += multiplier * self.__finite_difference(0, i)
			multiplier *= (t - i) / (i + 1)
		return result
			
	def __back_interpolation(self, x_input):
		t = (x_input - self.x[self.n - 1]) / (self.x[1] - self.x[0])
		multiplier = 1
		result = 0
		for i in range(self.n):
			result += multiplier * self.__finite_difference(self.n - 1 - i, i)
			multiplier *= (t + i) / (i + 1)
		return result
		
	def __Gauss_forward_interpolation(self, x_input, k):
		t = (x_input - self.x[k]) / (self.x[1] - self.x[0])
		multiplier = t
		result = self.y[k]
		for i in range(1, k + 1):
			result += multiplier * self.__finite_difference(k - i + 1, 2 * i - 1)
			multiplier *= (t - i) / (2 * i)
			result += multiplier * self.__finite_difference(k - i, 2 * i)
			multiplier *= (t ** 2 - i ** 2) / ((t - i) * (2 * i + 1))
		return result
		
	def __Gauss_back_interpolation(self, x_input, k):
		t = (x_input - self.x[k]) / (self.x[1] - self.x[0])
		multiplier = t
		result = self.y[k]
		for i in range(1, k + 1):
			result += multiplier * self.__finite_difference(k - i, 2 * i - 1)
			multiplier *= (t + i) / (2 * i)
			result += multiplier * self.__finite_difference(k - i, 2 * i)
			multiplier *= (t ** 2 - i ** 2) / ((t + i) * (2 * i + 1))
		return result
		
	def Newton_calculate(self, x_input):
		index = 0
		for i in range(1, self.n):
			if (abs(self.x[index] - x_input) > (abs(self.x[i] - x_input))):
				index = i
		if (x_input == self.x[index]):
			return self.y[index]
		if (self.n % 2 == 1):
			if (index < int((self.n - 1) / 2)):
				return self.__forward_interpolation(x_input)
			elif (index == int((self.n - 1) / 2)):
				if (x_input < self.x[index]):
					return self.__Gauss_back_interpolation(x_input, index)
				else:
					return self.__Gauss_forward_interpolation(x_input, index)
			else:
				return self.__back_interpolation(x_input)
		else:
			if (index < int(self.n / 2)):
				return self.__forward_interpolation(x_input)
			else:
				return self.__back_interpolation(x_input)
            
	def __finite_difference_error(self, i, k, y_error):
		if (k == 0):
			return y_error
		return self.__finite_difference_error(i + 1, k - 1, y_error) + self.__finite_difference_error(i, k - 1, y_error)
    
	def __forward_interpolation_error(self, x_input, y_error):
		t = (x_input - self.x[0]) / (self.x[1] - self.x[0])
		multiplier = 1
		result = 0
		for i in range(self.n):
			result += abs(multiplier) * self.__finite_difference_error(0, i, y_error)
			multiplier *= (t - i) / (i + 1)
		return result
			
	def __back_interpolation_error(self, x_input, y_error):
		t = (x_input - self.x[self.n - 1]) / (self.x[1] - self.x[0])
		multiplier = 1
		result = 0
		for i in range(self.n):
			result += abs(multiplier) * self.__finite_difference_error(self.n - 1 - i, i, y_error)
			multiplier *= (t + i) / (i + 1)
		return result
		
	def __Gauss_forward_interpolation_error(self, x_input, k, y_error):
		t = (x_input - self.x[k]) / (self.x[1] - self.x[0])
		multiplier = t
		result = y_error
		for i in range(1, k + 1):
			result += abs(multiplier) * self.__finite_difference_error(k - i + 1, 2 * i - 1, y_error)
			multiplier *= (t - i) / (2 * i)
			result += abs(multiplier) * self.__finite_difference_error(k - i, 2 * i, y_error)
			multiplier *= (t ** 2 - i ** 2) / ((t - i) * (2 * i + 1))
		return result
		
	def __Gauss_back_interpolation_error(self, x_input, k, y_error):
		t = (x_input - self.x[k]) / (self.x[1] - self.x[0])
		multiplier = t
		result = y_error
		for i in range(1, k + 1):
			result += abs(multiplier) * self.__finite_difference_error(k - i, 2 * i - 1, y_error)
			multiplier *= (t + i) / (2 * i)
			result += abs(multiplier) * self.__finite_difference_error(k - i, 2 * i, y_error)
			multiplier *= (t ** 2 - i ** 2) / ((t + i) * (2 * i + 1))
		return result
		
	def add_point(self, x_add, y_add):
		self.x = np.append(self.x, x_add)
		self.y = np.append(self.y, y_add)
		self.n += 1
			
	def Newton_error_calculate(self, x_input, y_error):
		index = 0
		for i in range(1, self.n):
			if (abs(self.x[index] - x_input) > (abs(self.x[i] - x_input))):
				index = i
		if (x_input == self.x[index]):
			return y_error
		if (self.n % 2 == 1):
			if (index < int((self.n - 1) / 2)):
				return self.__forward_interpolation_error(x_input, y_error)
			elif (index == int((self.n - 1) / 2)):
				if (x_input < self.x[index]):
					return self.__Gauss_back_interpolation_error(x_input, index, y_error)
				else:
					return self.__Gauss_forward_interpolation_error(x_input, index, y_error)
			else:
				return self.__back_interpolation_error(x_input, y_error)
		else:
			if (index < int(self.n / 2)):
				return self.__forward_interpolation_error(x_input, y_error)
			else:
				return self.__back_interpolation_error(x_input, y_error)
	
def main():
	x = np.array([0, 1.75, 3.5, 5.25, 7])
	y = np.array([0, -1.307, -2.211, -0.927, -0.871])
	polynomial = Newton(x, y, x.size)
	x_plot_table = np.linspace(0, 7, 50, dtype = float)
	y_plot_table = np.linspace(0, 0, 50, dtype = float)
	y_plot_table_of_original = np.linspace(0, 0, 50, dtype = float)
	x = np.append(x, 2.555)
	y = np.append(y, polynomial.Newton_calculate(2.555))
	for i in range(50):
		y_plot_table[i] = polynomial.Newton_calculate(x_plot_table[i])
		y_plot_table_of_original[i] = math.cos(x_plot_table[i]) - 2 ** (0.1 * x_plot_table[i])
	fig, ax = plt.subplots()
	plt.plot(x_plot_table, y_plot_table, 'b-', label = 'Newton')
	plt.plot(x_plot_table, y_plot_table_of_original, 'm--', label = 'Original')
	plt.plot(x, y, 'r*')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.annotate('f(x) ~ ' + str(y[y.size - 1]), xy=(x[x.size - 1], y[y.size - 1]), xytext=(x[x.size - 1], y[y.size - 1] - 0.32),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
	absolute_error = polynomial.Newton_error_calculate(2.555, 0.0005)
	plt.text(1.0, 0.0, 'Abs. error = ' + str(absolute_error))
	plt.text(1.0, -0.1, 'Rel. error = ' + str(-absolute_error / y[y.size - 1]))
	plt.legend(loc='upper right')
	plt.show()
	
if __name__ == '__main__':
    main()
