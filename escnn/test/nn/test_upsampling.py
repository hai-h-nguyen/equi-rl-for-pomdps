import unittest
from unittest import TestCase

from escnn.nn import *
from escnn.gspaces import *

import numpy as np


class TestUpsampling(TestCase):
    
    def test_cyclic_even_bilinear(self):
        g = rot2dOnR2(8)
        self.check_upsampling_scale(g, "bilinear")
        self.check_upsampling_size(g, "bilinear")

    def test_cyclic_odd_bilinear(self):
        g = rot2dOnR2(9)
        self.check_upsampling_scale(g, "bilinear")
        self.check_upsampling_size(g, "bilinear")

    def test_dihedral_even_bilinear(self):
        g = flipRot2dOnR2(8)
        self.check_upsampling_scale(g, "bilinear")
        self.check_upsampling_size(g, "bilinear")

    def test_dihedral_odd_bilinear(self):
        g = rot2dOnR2(9)
        self.check_upsampling_scale(g, "bilinear")
        self.check_upsampling_size(g, "bilinear")

    def test_so2_bilinear(self):
        g = rot2dOnR2(8)
        self.check_upsampling_scale(g, "bilinear")
        self.check_upsampling_size(g, "bilinear")

    def test_o2_bilinear(self):
        g = rot2dOnR2(8)
        self.check_upsampling_scale(g, "bilinear")
        self.check_upsampling_size(g, "bilinear")

    # "NEAREST" method is not equivariant!! As a result, all the following tests fail

    def test_cyclic_even_nearest(self):
        g = rot2dOnR2(8)
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_cyclic_odd_nearest(self):
        g = rot2dOnR2(9)
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_dihedral_even_nearest(self):
        g = flipRot2dOnR2(8)
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_dihedral_odd_nearest(self):
        g = rot2dOnR2(9)
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_so2_nearest(self):
        g = rot2dOnR2(8)
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def test_o2_nearest(self):
        g = rot2dOnR2(8)
        self.check_upsampling_scale(g, "nearest")
        self.check_upsampling_size(g, "nearest")

    def check_upsampling_scale(self, g, mode):
        for s in [2, 3, 5]:
            print(f"\nScale: {s}\n")
            for r in g.representations.values():
                r1 = FieldType(g, [r])
                ul = R2Upsampling(r1, mode=mode, scale_factor=s)
                ul.check_equivariance()

    def check_upsampling_size(self, g, mode):
        for s in [71, 129]:
            print(f"\nSize: {s}\n")
            for r in g.representations.values():
                r1 = FieldType(g, [r])
                ul = R2Upsampling(r1, mode=mode, size=s)
                ul.check_equivariance()

        
if __name__ == '__main__':
    unittest.main()
