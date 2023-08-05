import unittest
from unittest import TestCase

from escnn.nn import *
from escnn.gspaces import *

import numpy as np


class TestGroupedConv(TestCase):
    
    def test_cyclic(self):
        N = 4
        g = rot2dOnR2(N)

        groups = 5
        
        r1 = FieldType(g, list(g.representations.values()) * groups)
        r2 = FieldType(g, list(g.representations.values()) * groups)
        # r1 = FieldType(g, [g.trivial_repr])
        # r2 = FieldType(g, [g.regular_repr])
        
        s = 7
        sigma = None
        # fco = lambda r: 1. * r * np.pi
        fco = None

        cl = R2Conv(r1, r2, s, groups=groups,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)

        for _ in range(8):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_so2(self):
        N = 5
        g = rot2dOnR2(-1, N)
        groups = 5
        r1 = FieldType(g, list(g.representations.values()) * groups)
        r2 = FieldType(g, list(g.representations.values()) * groups)
    
        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2Conv(r1, r2, s, groups=groups,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)

        for _ in range(8):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_dihedral(self):
        N = 8
        g = flipRot2dOnR2(N, axis=np.pi/3)

        groups = 5
        r1 = FieldType(g, list(g.representations.values()) * groups)
        r2 = FieldType(g, list(g.representations.values()) * groups)
        # r1 = FieldType(g, [g.trivial_repr])
        # r2 = FieldType(g, [g.fibergroup.irrep(1, 0)])
        # r2 = FieldType(g, [irr for irr in g.fibergroup.irreps.values() if irr.size == 1])
        # r2 = FieldType(g, [g.regular_repr])
    
        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2Conv(r1, r2, s, groups=groups,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)

        for _ in range(8):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_o2(self):
        N = 5
        g = flipRot2dOnR2(-1, N)
        groups = 5
        r1 = FieldType(g, list(g.representations.values()) * groups)
        r2 = FieldType(g, list(g.representations.values()) * groups)
    
        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2Conv(r1, r2, s, groups=groups,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)

        for _ in range(8):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()

    def test_flip(self):
        # g = flip2dOnR2(axis=np.pi/3)
        g = flip2dOnR2(axis=np.pi/2)
        groups = 5
        r1 = FieldType(g, list(g.representations.values()) * groups)
        r2 = FieldType(g, list(g.representations.values()) * groups)
    
        s = 9
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2Conv(r1, r2, s, groups=groups,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)
        
        for _ in range(32):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion)
            cl.eval()
            cl.check_equivariance()


if __name__ == '__main__':
    unittest.main()
