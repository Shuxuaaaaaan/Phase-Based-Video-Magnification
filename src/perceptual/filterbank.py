from __future__ import division
import numpy as np
import scipy.misc as sc
import scipy.signal
from scipy.special import factorial

def visualize(coeff, normalize = True):
	M, N = coeff[1][0].shape
	Norients = len(coeff[1])
	out = np.zeros((M * 2 - coeff[-1].shape[0], Norients * N))

	currentx = 0
	currenty = 0
	for i in range(1, len(coeff[:-1])):
		for j in range(len(coeff[1])):
			tmp = coeff[i][j].real
			m,n = tmp.shape

			if normalize:
				tmp = 255 * tmp/tmp.max()

			tmp[m - 1, :] = 255
			tmp[:, n - 1] = 2555

			out[currentx : currentx + m, currenty : currenty + n] = tmp
			currenty += n
		currentx += coeff[i][0].shape[0]
		currenty = 0
	
	m,n = coeff[-1].shape
	out[currentx : currentx+m, currenty : currenty+n] = 255 * coeff[-1]/coeff[-1].max()

	out[0,:] = 255
	out[:, 0] = 255

	return out

class Steerable:
	def __init__(self, height = 5, use_cuda = False):
		"""
		height is the total height, including highpass and lowpass
		"""
		self.nbands = 4
		self.height = height
		self.isSample = True
		self.use_cuda = use_cuda
		if self.use_cuda:
			import cupy as cp
			import cupyx.scipy.fft as cp_fft
			self.xp = cp
			self.fft = cp_fft
		else:
			self.xp = np
			self.fft = np.fft

	def buildSCFpyr(self, im):
		# im is assumed to be already an xp array here (np or cp)
		assert len(im.shape) == 2, 'Input image must be grayscale'

		M, N = im.shape
		log_rad, angle = self.base(M, N)
		Xrcos, Yrcos = self.rcosFn(1, -0.5)
		Yrcos = self.xp.sqrt(Yrcos)
		YIrcos = self.xp.sqrt(1 - Yrcos*Yrcos)

		lo0mask = self.pointOp(log_rad, YIrcos, Xrcos)
		hi0mask = self.pointOp(log_rad, Yrcos, Xrcos)

		imdft = self.fft.fftshift(self.fft.fft2(im))
		lo0dft = imdft * lo0mask

		coeff = self.buildSCFpyrlevs(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height - 1)

		hi0dft = imdft * hi0mask
		hi0 = self.fft.ifft2(self.fft.ifftshift(hi0dft))

		coeff.insert(0, hi0.real)

		return coeff

	def getlist(self, coeff):
		straight = [bands for scale in coeff[1:-1] for bands in scale]
		straight = [coeff[0]] + straight + [coeff[-1]]
		return straight

	def buildSCFpyrlevs(self, lodft, log_rad, angle, Xrcos, Yrcos, ht):
		if (ht <=1):
			lo0 = self.fft.ifft2(self.fft.ifftshift(lodft))
			coeff = [lo0.real]
		
		else:
			Xrcos = Xrcos - 1

			# ==================== Orientation bandpass =======================
			himask = self.pointOp(log_rad, Yrcos, Xrcos)

			lutsize = 1024
			Xcosn = self.xp.pi * self.xp.arange(-(2*lutsize+1),(lutsize+2))/lutsize
			order = self.nbands - 1
			const = self.xp.power(2, 2*order) * self.xp.square(factorial(order)) / (self.nbands * factorial(2*order))

			alpha = (Xcosn + self.xp.pi) % (2*self.xp.pi) - self.xp.pi
			Ycosn = 2*self.xp.sqrt(const) * self.xp.power(self.xp.cos(Xcosn), order) * (self.xp.abs(alpha) < self.xp.pi/2)

			orients = []

			for b in range(self.nbands):
				anglemask = self.pointOp(angle, Ycosn, Xcosn + self.xp.pi*b/self.nbands)
				banddft = self.xp.power(complex(0,-1), self.nbands - 1) * lodft * anglemask * himask
				band = self.fft.ifft2(self.fft.ifftshift(banddft))
				orients.append(band)

			# ================== Subsample lowpass ============================
			dims = self.xp.array(lodft.shape)
			
			lostart = self.xp.ceil((dims+0.5)/2) - self.xp.ceil((self.xp.ceil((dims-0.5)/2)+0.5)/2)
			loend = lostart + self.xp.ceil((dims-0.5)/2)

			lostart = lostart.astype(int)
			loend = loend.astype(int)

			log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
			angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
			lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
			YIrcos = self.xp.abs(self.xp.sqrt(1 - Yrcos*Yrcos))
			lomask = self.pointOp(log_rad, YIrcos, Xrcos)

			lodft = lomask * lodft

			coeff = self.buildSCFpyrlevs(lodft, log_rad, angle, Xrcos, Yrcos, ht-1)
			coeff.insert(0, orients)

		return coeff

	def reconSCFpyrLevs(self, coeff, log_rad, Xrcos, Yrcos, angle):

		if (len(coeff) == 1):
			return self.fft.fftshift(self.fft.fft2(coeff[0]))

		else:

			Xrcos = Xrcos - 1
    		
    		# ========================== Orientation residue==========================
			himask = self.pointOp(log_rad, Yrcos, Xrcos)

			lutsize = 1024
			Xcosn = self.xp.pi * self.xp.arange(-(2*lutsize+1),(lutsize+2))/lutsize
			order = self.nbands - 1
			const = self.xp.power(2, 2*order) * self.xp.square(factorial(order)) / (self.nbands * factorial(2*order))
			Ycosn = self.xp.sqrt(const) * self.xp.power(self.xp.cos(Xcosn), order)

			orientdft = self.xp.zeros(coeff[0][0].shape, dtype=complex)

			for b in range(self.nbands):
				anglemask = self.pointOp(angle, Ycosn, Xcosn + self.xp.pi* b/self.nbands)
				banddft = self.fft.fftshift(self.fft.fft2(coeff[0][b]))
				orientdft = orientdft + self.xp.power(complex(0,1), order) * banddft * anglemask * himask

			# ============== Lowpass component are upsampled and convoluted ============
			dims = self.xp.array(coeff[0][0].shape)
			
			lostart = (self.xp.ceil((dims+0.5)/2) - self.xp.ceil((self.xp.ceil((dims-0.5)/2)+0.5)/2)).astype(self.xp.int32)
			loend = lostart + self.xp.ceil((dims-0.5)/2).astype(self.xp.int32) 
			lostart = lostart.astype(int)
			loend = loend.astype(int)

			nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
			nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
			YIrcos = self.xp.sqrt(self.xp.abs(1 - Yrcos * Yrcos))
			lomask = self.pointOp(nlog_rad, YIrcos, Xrcos)

			nresdft = self.reconSCFpyrLevs(coeff[1:], nlog_rad, Xrcos, Yrcos, nangle)

			res = self.fft.fftshift(self.fft.fft2(nresdft))

			resdft = self.xp.zeros(dims.tolist(), dtype='complex128')
			resdft[lostart[0]:loend[0], lostart[1]:loend[1]] = nresdft * lomask

			return resdft + orientdft

	def reconSCFpyr(self, coeff):

		if (self.nbands != len(coeff[1])):
			raise Exception("Unmatched number of orientations")

		M, N = coeff[0].shape
		log_rad, angle = self.base(M, N)

		Xrcos, Yrcos = self.rcosFn(1, -0.5)
		Yrcos = self.xp.sqrt(Yrcos)
		YIrcos = self.xp.sqrt(self.xp.abs(1 - Yrcos*Yrcos))

		lo0mask = self.pointOp(log_rad, YIrcos, Xrcos)
		hi0mask = self.pointOp(log_rad, Yrcos, Xrcos)

		tempdft = self.reconSCFpyrLevs(coeff[1:], log_rad, Xrcos, Yrcos, angle)

		hidft = self.fft.fftshift(self.fft.fft2(coeff[0]))
		outdft = tempdft * lo0mask + hidft * hi0mask

		return self.fft.ifft2(self.fft.ifftshift(outdft)).real.astype(int)


	def base(self, m, n):
		
		x = self.xp.linspace(-(m // 2)/(m / 2), (m // 2)/(m / 2) - (1 - m % 2)*2/m , num = m)
		y = self.xp.linspace(-(n // 2)/(n / 2), (n // 2)/(n / 2) - (1 - n % 2)*2/n , num = n)

		xv, yv = self.xp.meshgrid(y, x)

		angle = self.xp.arctan2(yv, xv)

		rad = self.xp.sqrt(xv**2 + yv**2)
		rad[m//2][n//2] = rad[m//2][n//2 - 1]
		log_rad = self.xp.log2(rad)

		return log_rad, angle

	def rcosFn(self, width, position):
		N = 256
		X = self.xp.pi * self.xp.arange(-N-1, 2)/2/N

		Y = self.xp.cos(X)**2
		Y[0] = Y[1]
		Y[N+2] = Y[N+1]

		X = position + 2*width/self.xp.pi*(X + self.xp.pi/4)
		return X, Y

	def pointOp(self, im, Y, X):
		out = self.xp.interp(im.flatten(), X, Y)
		return self.xp.reshape(out, im.shape)

class SteerableNoSub(Steerable):

	def buildSCFpyrlevs(self, lodft, log_rad, angle, Xrcos, Yrcos, ht):
		if (ht <=1):
			lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
			coeff = [lo0.real]
		
		else:
			Xrcos = Xrcos - 1

			# ==================== Orientation bandpass =======================
			himask = self.pointOp(log_rad, Yrcos, Xrcos)

			lutsize = 1024
			Xcosn = np.pi * np.array(range(-(2*lutsize+1),(lutsize+2)))/lutsize
			order = self.nbands - 1
			const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))

			alpha = (Xcosn + np.pi) % (2*np.pi) - np.pi
			Ycosn = 2*np.sqrt(const) * np.power(np.cos(Xcosn), order) * (np.abs(alpha) < np.pi/2)

			orients = []

			for b in range(self.nbands):
				anglemask = self.pointOp(angle, Ycosn, Xcosn + np.pi*b/self.nbands)
				banddft = np.power(complex(0,-1), self.nbands - 1) * lodft * anglemask * himask
				band = np.fft.ifft2(np.fft.ifftshift(banddft))
				orients.append(band)

			# ================== Subsample lowpass ============================
			lostart = (0, 0)
			loend = lodft.shape

			log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
			angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
			lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
			YIrcos = np.abs(np.sqrt(1 - Yrcos*Yrcos))
			lomask = self.pointOp(log_rad, YIrcos, Xrcos)

			lodft = lomask * lodft

			coeff = self.buildSCFpyrlevs(lodft, log_rad, angle, Xrcos, Yrcos, ht-1)
			coeff.insert(0, orients)

		return coeff
