# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 18:19:12 2015

@author: jkooij
"""

import numpy as np

class Pyramid2arr:
    '''Class for converting a pyramid to/from a 1d array'''
    
    def __init__(self, steer, coeff=None):
        """
        Initialize class with sizes from pyramid coeff
        """
        self.levels = range(1, steer.height-1)
        self.bands = range(steer.nbands)
        
        self._indices = None
        if coeff is not None:
            self.init_coeff(coeff)

    def init_coeff(self, coeff):       
        shapes = [coeff[0].shape]        
        for lvl in self.levels:
            for b in self.bands:
                shapes.append( coeff[lvl][b].shape )             
        shapes.append(coeff[-1].shape)

        # compute the total sizes        
        sizes = [np.prod(shape) for shape in shapes]
        
        # precompute indices of each band
        offsets = np.cumsum([0] + sizes)
        self._indices = list(zip(offsets[:-1], offsets[1:], shapes))

    def p2a(self, coeff):
        """
        Convert pyramid as a 1d Array
        """
        
        if self._indices is None:
            self.init_coeff(coeff)
            
        # Detect backend from the input coeff array
        xp = np
        if type(coeff[0]).__module__.startswith('cupy'):
            import cupy as cp
            xp = cp
        
        bandArray = xp.hstack([ coeff[lvl][b].ravel() for lvl in self.levels for b in self.bands ])
        bandArray = xp.hstack((coeff[0].ravel(), bandArray, coeff[-1].ravel()))

        return bandArray        
        
       
    def a2p(self, bandArray):
        """
        Convert 1d array back to Pyramid
        """
        
        assert self._indices is not None, 'Initialize Pyramid2arr first with init_coeff() or p2a()'

        xp = np
        if type(bandArray).__module__.startswith('cupy'):
            import cupy as cp
            xp = cp

        # create iterator that convert array to images
        it = (xp.reshape(bandArray[istart:iend], size) for (istart,iend,size) in self._indices)
        
        coeffs = [next(it)]
        for lvl in self.levels:
            coeffs.append([next(it) for band in self.bands])
        coeffs.append(next(it))

        return coeffs

