import numpy as np
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
from ldpc.mod2 import row_basis, nullspace, rank
import itertools

def subdivide(rhombs):
    result = []
    scale_factor = 0.5/(1+np.cos(np.pi/7)+np.cos(2*np.pi/7))

    for ctg, A, B, C, D in rhombs:
        base_vector = B-A
        if ctg == 0:  # pi/7 rhomb - 1
            l1 = scale_factor*base_vector*np.exp(1j*2*np.pi/7)
            l2 = scale_factor*base_vector*np.exp(1j*np.pi/7)
            l3 = scale_factor*base_vector
            l4 = scale_factor*base_vector*np.exp(-1j*np.pi/7)
            l5 = scale_factor*base_vector*np.exp(-1j*2*np.pi/7)
            l6 = scale_factor*base_vector*np.exp(-1j*3*np.pi/7)

            P1 = A + l3
            P2 = A + l4
            P4 = P1 + l4
            P3 = P4 + l2
            P5 = P4 + l5
            P6 = P3 + l5
            P7 = P6 + l1
            P8 = P6 + l6
            P9 = P6 + l4
            P10 = P9 + l1
            P11 = D + l1
            P12 = P10 + l3
            P13 = P12 + l6
            P14 = D + l3
            P15 = P13 + l2
            P16 = P13 + l5
            P17 = P16 + l2
            P18 = P17 + l3
            P19 = P17 + l4
            P20 = P1 + l2
            P21 = P2 + l5
            P22 = P15 + l3
            P23 = P16 + l4
            P24 = P3 + l1
            P25 = P5 + l6
            P26 = P12 + l2
            P27 = P14 + l5
            
            F1 = (0, P4, P2, A, P1)
            F2 = (2, P20, P3, P4, P1)
            F3 = (2, P4, P5, P21, P2)
            F4 = (5, P6, P5, P4, P3)
            F5 = (5, P6, P3, P24, P7)
            F6 = (5, P25, P5, P6, P8)
            F7 = (5, P6, P7, P10, P9)
            F8 = (3, P8, P6, P9, D)
            F9 = (1, P7, B, P12, P10)
            F10 = (3, P9, P10, P11, D)
            F11 = (5, P10, P12, P13, P11)
            F12 = (3, P11, P13, P14, D)
            F13 = (5, P26, P15, P13, P12)
            F14 = (5, P13, P16, P27, P14)
            F15 = (5, P17, P16, P13, P15)
            F16 = (2, P17, P15, P22, P18)
            F17 = (2, P23, P16, P17, P19)
            F18 = (0, C, P19, P17, P18)
        
            result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
                          F11, F12, F13, F14, F15, F16, F17, F18]
        elif ctg == 1:  # pi/7 rhomb - 2
            l0 = scale_factor*base_vector
            l1 = scale_factor*base_vector*np.exp(1j*np.pi/7)
            l2 = scale_factor*base_vector*np.exp(-1j*np.pi/7)
            l3 = scale_factor*base_vector*np.exp(1j*2*np.pi/7)
            l4 = scale_factor*base_vector*np.exp(-1j*2*np.pi/7)
            l5 = scale_factor*base_vector*np.exp(1j*3*np.pi/7)
            l6 = scale_factor*base_vector*np.exp(-1j*3*np.pi/7)

            P1 = A + l1
            P2 = A + l2
            P3 = P2 + l6
            P4 = P1 + l2
            P5 = P4 + l6
            P6 = P4 + l0
            P7 = P6 + l6
            P8 = P5 + l4
            P9 = P6 + l5
            P10 = P9 + l6
            P11 = P8 + l0
            P12 = P10 + l1
            P13 = P10 + l2
            P14 = P7 + l2
            P15 = P14 + l1
            P16 = B + l0
            P17 = B + l4
            P18 = D + l1
            P19 = D + l0
            P20 = P17 + l0
            P21 = P20 + l3
            P23 = P18 + l0
            P22 = P23 + l3
            P24 = P19 + l2
            P25 = P22 + l2
            P26 = P23 + l2
            P27 = P26 + l4
            P28 = P25 + l0
            P29 = P25 + l4

            F1 = (3, P1, P4, P2, A)
            F2 = (5, P4, P5, P3, P2)
            F3 = (5, P7, P5, P4, P6)
            F4 = (0, P7, P6, P9, P10)
            F5 = (2, P8, P5, P7, P11)
            F6 = (6, P14, P7, P10, P13)
            F7 = (0, D, P11, P7, P14)
            F8 = (4, P13, P10, P12, B)
            F9 = (4, P15, P14, P13, B)
            F10 = (6, P18, D, P14, P15)
            F11 = (4, P17, P18, P15, B)
            F12 = (0, P23, P19, D, P18)
            F13 = (4, P16, P20, P17, B)
            F14 = (6, P20, P23, P18, P17)
            F15 = (0, P21, P22, P23, P20)
            F16 = (2, P23, P26, P24, P19)
            F17 = (5, P25, P26, P23, P22)
            F18 = (5, P27, P26, P25, P29)
            F19 = (3, P29, P25, P28, C)

            result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
                            F11, F12, F13, F14, F15, F16, F17, F18, F19]
        elif ctg == 2:  # 2pi/7 rhomb - 1
            l1 = scale_factor*base_vector*np.exp(-1j*np.pi/7)
            l2 = scale_factor*base_vector*np.exp(-1j*2*np.pi/7)
            l3 = scale_factor*base_vector
            l4 = scale_factor*base_vector*np.exp(-1j*3*np.pi/7)
            l5 = scale_factor*base_vector*np.exp(1j*np.pi/7)
            l6 = scale_factor*base_vector*np.exp(-1j*4*np.pi/7)
            l7 = scale_factor*base_vector*np.exp(1j*2*np.pi/7)
            l8 = scale_factor*base_vector*np.exp(-1j*5*np.pi/7)

            P1 = A + l8
            P2 = P1 - l5
            P3 = P2 + l6
            P4 = P3 - l3
            P5 = P4 + l4
            P6 = A + l3
            P7 = A + l1
            P8 = A + l2
            P9 = P1 + l2
            P10 = P1 + l6
            P11 = P10 + l2
            P12 = P10 + l4
            P13 = P3 + l4
            P14 = P13 + l8
            P15 = P6 + l5
            P16 = P8 + l1
            P17 = P9 + l1
            P18 = P11 + l1
            P19 = P11 + l4
            P20 = P19 - l5
            P21 = P20 + l8
            P23 = P15 + l1
            P22 = P23 + l7
            P38 = P6 + l1
            P24 = P38 + l2
            P25 = P24 + l8
            P26 = P25 + l6
            P27 = P18 + l4
            P28 = P20 + l1
            P29 = P22 + l2
            P30 = P29 + l8
            P31 = P24 + l4
            P32 = P31 + l8
            P33 = B + l8
            P34 = P33 + l4
            P35 = P30 + l4
            P36 = P35 + l6
            P37 = P31 + l6

            F1 = (0, P38, P7, A, P6)
            F2 = (0, A, P7, P16, P8)
            F3 = (5, P9, P1, A, P8)
            F4 = (2, P9, P11, P10, P1)
            F5 = (2, P10, P3, P2, P1)
            F6 = (5, P10, P12, P13, P3)
            F7 = (5, P4, P3, P13, P5)
            F8 = (3, P5, P13, P14, D)
            F9 = (2, P15, P23, P38, P6)
            F10 = (2, P38, P24, P16, P7)
            F11 = (5, P9, P8, P16, P17)
            F12 = (5, P9, P17, P18, P11)
            F13 = (0, P10, P11, P19, P12)
            F14 = (5, P13, P12, P19, P20)
            F15 = (5, P13, P20, P21, P14)
            F16 = (5, P30, P23, P22, P29)
            F17 = (5, P30, P24, P38, P23)
            F18 = (2, P16, P24, P25, P17)
            F19 = (5, P25, P26, P18, P17)
            F20 = (2, P19, P11, P18, P27)
            F21 = (2, P28, P20, P19, P27)
            F22 = (3, P33, P30, P29, B)
            F23 = (5, P34, P35, P30, P33)
            F24 = (5, P31, P24, P30, P35)
            F25 = (2, P25, P24, P31, P32)
            F26 = (2, P31, P35, P36, P37)
            F27 = (0, P25, P32, C, P26)
            F28 = (0, C, P32, P31, P37)
            F29 = (5, C, P27, P18, P26)

            result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
                            F11, F12, F13, F14, F15, F16, F17, F18, F19,
                            F20, F21, F22, F23, F24, F25, F26, F27, F28, F29]   
        elif ctg == 3:  # 2pi/7 rhomb - 2
            l1 = scale_factor*base_vector*np.exp(-1j*np.pi/7)
            l2 = scale_factor*base_vector*np.exp(-1j*2*np.pi/7)
            l3 = scale_factor*base_vector
            l4 = scale_factor*base_vector*np.exp(-1j*3*np.pi/7)
            l5 = scale_factor*base_vector*np.exp(1j*np.pi/7)
            l6 = scale_factor*base_vector*np.exp(-1j*4*np.pi/7)
            l7 = scale_factor*base_vector*np.exp(1j*2*np.pi/7)
            l8 = scale_factor*base_vector*np.exp(-1j*5*np.pi/7)

            P1 = A - l5
            P2 = A + l6
            P3 = P1 + l6
            P4 = P3 + l8
            P5 = P4 - l1
            P6 = P5 + l2
            P7 = P6 - l5
            P8 = P6 + l6
            P10 = A + l3
            P9 = P10 + l7
            P11 = P9 + l2
            P12 = P10 + l2
            P13 = A + l2
            P14 = P12 + l6
            P15 = P13 + l6
            P16 = P3 + l2
            P17 = P15 + l8
            P18 = P16 + l8
            P19 = P18 + l6
            P20 = P18 + l4
            P21 = P19 + l4
            P22 = P17 + l3
            P23 = P18 + l3
            P24 = P11 + l5
            P25 = P24 + l1
            P26 = P11 + l1
            P27 = P12 + l1
            P28 = P14 + l1
            P29 = P22 + l1
            P30 = P22 + l4
            P31 = P30 - l5
            P32 = B - l5
            P33 = P32 + l8
            P34 = B + l8
            P35 = P34 + l6
            P36 = P33 + l6
            P37 = P36 + l4
            P38 = P28 + l4
            P39 = P31 + l1
            P40 = D + l1

            F1 = (3, P2, P3, P1, A)
            F2 = (3, P13, P15, P2, A)
            F3 = (3, P10, P12, P13, A)
            F4 = (5, P15, P16, P3, P2)
            F5 = (5, P3, P16, P18, P4)
            F6 = (0, P5, P4, P18, P6)
            F7 = (6, P6, P18, P19, P8)
            F8 = (4, P7, P6, P8, D)
            F9 = (4, P8, P19, P40, D)
            F10 = (5, P9, P11, P12, P10)
            F11 = (5, P12, P14, P15, P13)
            F12 = (2, P15, P14, P22, P17)
            F13 = (5, P18, P23, P31, P20)
            F14 = (2, P26, P11, P24, P25)
            F15 = (5, P12, P11, P26, P27)
            F16 = (5, P28, P14, P12, P27)
            F17 = (5, P22, P14, P28, P29)
            F18 = (5, P31, P23, P22, P30)
            F19 = (0, P26, P25, B, P32)
            F20 = (0, B, P34, P33, P32)
            F21 = (2, P33, P27, P26, P32)
            F22 = (2, P35, P36, P33, P34)
            F23 = (5, P33, P36, P28, P27)
            F24 = (5, P28, P36, P37, P38)
            F25 = (3, P29, P28, P38, C)
            F26 = (3, P30, P22, P29, C)
            F27 = (3, P39, P31, P30, C)
            F28 = (0, P15, P17, P18, P16)
            F29 = (0, P18, P17, P22, P23)
            F30 = (0, P18, P20, P21, P19)

            result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
                            F11, F12, F13, F14, F15, F16, F17, F18, F19,
                            F20, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30]
        elif ctg == 4:  # 2pi/7 rhomb - 3
            l1 = scale_factor*base_vector*np.exp(-1j*np.pi/7)
            l2 = scale_factor*base_vector*np.exp(-1j*2*np.pi/7)
            l3 = scale_factor*base_vector
            l4 = scale_factor*base_vector*np.exp(-1j*3*np.pi/7)
            l5 = scale_factor*base_vector*np.exp(1j*np.pi/7)
            l6 = scale_factor*base_vector*np.exp(-1j*4*np.pi/7)
            l7 = scale_factor*base_vector*np.exp(1j*2*np.pi/7)
            l8 = scale_factor*base_vector*np.exp(-1j*5*np.pi/7)

            P1 = A + l8
            P2 = P1 - l3
            P3 = P2 + l4
            P4 = P3 + l8
            P5 = P4 - l5
            P6 = A + l5
            P8 = P6 + l1
            P7 = P8 - l6
            P9 = A + l1
            P10 = A + l4
            P11 = P1 + l4
            P12 = P3 + l1
            P13 = P4 + l1
            P14 = P4 + l6
            P15 = P14 + l1
            P16 = D + l1
            P17 = P7 + l4
            P18 = P8 + l4
            P19 = P9 + l4
            P20 = P19 + l8
            P21 = P20 + l8
            P22 = P21 + l6
            P23 = P17 + l3
            P24 = P23 + l6
            P25 = P24 + l8
            P26 = P20 + l3
            P27 = P26 + l6
            P28 = P20 + l6
            P29 = P28 + l2
            P30 = P22 + l2
            P31 = P23 + l5
            P32 = P23 + l1
            P33 = B + l6
            P34 = P32 + l6
            P35 = P34 + l8
            P36 = P35 + l2
            P37 = P25 + l2
            P38 = P26 + l2
            P39 = C + l5
            P40 = P18 + l8

            F1 = (6, P11, P3, P2, P1)
            F2 = (6, P4, P3, P12, P13)
            F3 = (6, P15, P14, P4, P13)
            F4 = (2, P5, P4, P14, D)
            F5 = (2, P14, P15, P16, D)
            F6 = (4, P10, P11, P1, A)
            F7 = (4, P9, P19, P10, A)
            F8 = (4, P6, P8, P9, A)
            F9 = (0, P20, P12, P3, P11)
            F10 = (3, P12, P20, P21, P13)
            F11 = (6, P21, P22, P15, P13)
            F12 = (5, P18, P17, P23, P24)
            F13 = (2, P18, P24, P25, P40)
            F14 = (0, P18, P40, P20, P19)
            F15 = (0, P20, P40, P25, P26)
            F16 = (6, P28, P20, P26, P27)
            F17 = (0, P22, P21, P20, P28)
            F18 = (6, P30, P22, P28, P29)
            F19 = (5, P23, P32, P34, P24)
            F20 = (3, P32, P23, P31, B)
            F21 = (3, P33, P34, P32, B)
            F22 = (5, P34, P35, P25, P24)
            F23 = (0, P25, P35, P36, P37)
            F24 = (6, P26, P25, P37, P38)
            F25 = (4, P38, P37, P39, C)
            F26 = (4, P27, P26, P38, C)
            F27 = (4, P29, P28, P27, C)
            F28 = (0, P7, P17, P18, P8)
            F29 = (6, P8, P18, P19, P9)
            F30 = (6, P19, P20, P11, P10)

            result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
                        F11, F12, F13, F14, F15, F16, F17, F18, F19, F20,
                        F21, F22, F23, F24, F25, F26, F27, F28, F29, F30]
        elif ctg == 5:  # 3pi/7 rhomb - 1
            l0 = scale_factor*base_vector*np.exp(1j*2*np.pi/7)
            l1 = scale_factor*base_vector*np.exp(1j*np.pi/7)
            l2 = scale_factor*base_vector
            l3 = scale_factor*base_vector*np.exp(-1j*np.pi/7)
            l4 = scale_factor*base_vector*np.exp(-1j*2*np.pi/7)
            l5 = scale_factor*base_vector*np.exp(-1j*3*np.pi/7)
            l6 = scale_factor*base_vector*np.exp(-1j*4*np.pi/7)

            P1 = A + l5
            P2 = A + l4
            P3 = A + l3
            P4 = A + l2
            P5 = P1 + l6
            P6 = P1 + l4
            P7 = P2 + l3
            P8 = P3 + l2
            P9 = P4 + l1
            P10 = P5 + l4
            P11 = P6 + l3
            P12 = P7 + l2
            P13 = P8 + l1
            P14 = P10 - l0
            P15 = P10 + l3
            P16 = P11 + l2
            P17 = P12 + l1
            P18 = P13 + l0
            P19 = P14 + l3
            P20 = P18 + l4
            P21 = D + l0
            P22 = P15 + l2
            P23 = P22 + l1
            P24 = P16 + l1
            P25 = P17 + l2
            P26 = D + l2
            P27 = P21 + l2
            P28 = P27 + l1
            P29 = P23 + l2
            P30 = P24 + l2
            P31 = B + l5
            P32 = P26 + l4
            P33 = P27 + l4
            P34 = P30 + l3
            P35 = P31 + l3
            P36 = P33 + l3
            P37 = P28 + l4
            P38 = P28 + l2
            P39 = P29 + l3
            P40 = P34 + l4
            P41 = P37 + l3
            P42 = P37 + l2
            P43 = P38 + l3
            P44 = P39 + l4
            
            F1 = (0, P6, P1, A, P2)
            F2 = (0, A, P3, P7, P2)
            F3 = (0, P8, P3, A, P4)
            F4 = (2, P6, P10, P5, P1)
            F5 = (2, P7, P11, P6, P2)
            F6 = (2, P8, P12, P7, P3)
            F7 = (2, P9, P13, P8, P4)
            F8 = (5, P15, P10, P6, P11)
            F9 = (5, P7, P12, P16, P11)
            F10 = (5, P17, P12, P8, P13)
            F11 = (5, P14, P10, P15, P19)
            F12 = (5, P15, P11, P16, P22)
            F13 = (5, P16, P12, P17, P24)
            F14 = (5, P17, P13, P18, P20)
            F15 = (3, P19, P15, P21, D)
            F16 = (5, P15, P22, P27, P21)
            F17 = (2, P16, P24, P23, P22)
            F18 = (5, P30, P24, P17, P25)
            F19 = (3, P25, P17, P20, B)
            F20 = (3, P21, P27, P26, D)
            F21 = (5, P23, P28, P27, P22)
            F22 = (5, P30, P29, P23, P24)
            F23 = (3, P31, P30, P25, B)
            F24 = (5, P27, P33, P32, P26)
            F25 = (5, P37, P33, P27, P28)
            F26 = (5, P23, P29, P38, P28)
            F27 = (5, P39, P29, P30, P34)
            F28 = (5, P35, P34, P30, P31)
            F29 = (2, P36, P33, P37, P41)
            F30 = (2, P37, P28, P38, P42)
            F31 = (2, P38, P29, P39, P43)
            F32 = (2, P39, P34, P40, P44)
            F33 = (0, C, P41, P37, P42)
            F34 = (0, P38, P43, C, P42)
            F35 = (0, C, P43, P39, P44)

            result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
                            F11, F12, F13, F14, F15, F16, F17, F18, F19,
                            F20, F21, F22, F23, F24, F25, F26, F27, F28,
                            F29, F30, F31, F32, F33, F34, F35]
        elif ctg == 6:  # 3pi/7 rhomb - 1
            l0 = scale_factor*base_vector*np.exp(1j*2*np.pi/7)
            l1 = scale_factor*base_vector*np.exp(1j*np.pi/7)
            l2 = scale_factor*base_vector
            l3 = scale_factor*base_vector*np.exp(-1j*np.pi/7)
            l4 = scale_factor*base_vector*np.exp(-1j*2*np.pi/7)
            l5 = scale_factor*base_vector*np.exp(-1j*3*np.pi/7)
            l6 = scale_factor*base_vector*np.exp(-1j*4*np.pi/7)

            P1 = B - l2
            P2 = B - l1
            P3 = B - l0
            P4 = B + l6
            P5 = B + l5
            P6 = P1 - l3
            P7 = P2 - l2
            P8 = P3 - l1
            P9 = P4 - l0
            P10 = P5 + l6
            P11 = P5 + l4
            P12 = P6 - l1
            P13 = P7 - l0
            P14 = P8 + l6
            P15 = P9 + l5
            P16 = P10 + l4
            P17 = P12 - l4
            P18 = P12 - l0
            P19 = P13 + l6
            P20 = P14 + l5
            P21 = P15 + l4
            P22 = P16 + l3
            P23 = P17 - l0
            P24 = P21 + l3
            P25 = A + l4
            P26 = P18 + l6
            P27 = P19 + l5
            P28 = P20 + l4
            P29 = P21 + l5
            P30 = A + l6
            P31 = P25 + l6
            P32 = P26 + l5
            P33 = P27 + l4
            P34 = P28 + l5
            P35 = P34 + l3
            P36 = P32 + l4
            P37 = P31 + l5
            P38 = P32 - l1
            P39 = P36 - l1
            P40 = P36 + l6
            P41 = P33 + l6
            P42 = P33 + l5
            P43 = P37 - l1
            P44 = D - l4
            P45 = D + l3
            P46 = P41 + l5

            F1 = (0, P7, P1, B, P2)
            F2 = (0, B, P3, P8, P2)
            F3 = (0, P9, P3, B, P4)
            F4 = (0, B, P5, P10, P4)
            F5 = (2, P7, P12, P6, P1)
            F6 = (2, P8, P13, P7, P2)
            F7 = (2, P9, P14, P8, P3)
            F8 = (2, P10, P15, P9, P4)
            F9 = (2, P11, P16, P10, P5)
            F10 = (5, P18, P12, P7, P13)
            F11 = (5, P8, P14, P19, P13)
            F12 = (5, P20, P14, P9, P15)
            F13 = (5, P10, P16, P21, P15)
            F14 = (5, P17, P12, P18, P23)
            F15 = (5, P19, P26, P18, P13)
            F16 = (5, P19, P14, P20, P27)
            F17 = (5, P21, P28, P20, P15)
            F18 = (3, P23, P18, P25, A)
            F19 = (5, P18, P26, P31, P25)
            F20 = (2, P32, P26, P19, P27)
            F21 = (2, P20, P28, P33, P27)
            F22 = (5, P34, P28, P21, P29)
            F23 = (3, P29, P21, P24, C)
            F24 = (3, P25, P31, P30, A)
            F25 = (5, P31, P26, P32, P37)
            F26 = (5, P33, P28, P34, P42)
            F27 = (3, P35, P34, P29, C)
            F28 = (0, P43, P37, P32, P38)
            F29 = (6, P38, P32, P36, P39)
            F30 = (6, P36, P33, P41, P40)
            F31 = (0, P33, P42, P46, P41)
            F32 = (4, P44, P38, P39, D)
            F33 = (4, P39, P36, P40, D)
            F34 = (4, P40, P41, P45, D)
            F35 = (5, P21, P16, P22, P24)
            F36 = (0, P32, P27, P33, P36)

            result += [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
                            F11, F12, F13, F14, F15, F16, F17, F18, F19,
                            F20, F21, F22, F23, F24, F25, F26, F27, F28,
                            F29, F30, F31, F32, F33, F34, F35, F36]
            
    return result

def close(a, b):
    return np.linalg.norm(a-b) < 1e-5

def get_vertices(rhombs):
    vertices = []
    for rhomb in rhombs:
        vertices.append(rhomb[1])
        vertices.append(rhomb[2])
        vertices.append(rhomb[3])
        vertices.append(rhomb[4])
    vertices_new = []
    for v in vertices:
        if not any(close(v, v2) for v2 in vertices_new):
            vertices_new.append(v)
    return vertices_new

def get_edges(faces, vertices):
    def vertices_on_face(face, vertices):
        vs_on_f = [face[0]] # color
        for v in vertices:
            if close(face[1], v):
                vs_on_f.append(v)
        for v in vertices:
            if close(face[2], v):
                vs_on_f.append(v)
        for v in vertices:
            if close(face[3], v):
                vs_on_f.append(v)
        for v in vertices:
            if close(face[4], v):
                vs_on_f.append(v)
        return vs_on_f

    edges = []
    for face in faces:
        vs_on_f = vertices_on_face(face, vertices)
        edges.append((vs_on_f[1], vs_on_f[2]))
        edges.append((vs_on_f[2], vs_on_f[3]))
        edges.append((vs_on_f[3], vs_on_f[4]))
        edges.append((vs_on_f[4], vs_on_f[1]))
    return edges

def draw_schaad(faces, vertices):
    vertices_pos = [np.array([vertex.real, vertex.imag]) for vertex in vertices]
    edges = get_edges(faces, vertices)
    fig, ax = plt.subplots()
    ax.scatter(np.array(vertices_pos)[:,0], np.array(vertices_pos)[:,1], marker='o', c='b')
    edges = get_edges(faces, vertices)
    for edge in edges:
        ax.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], color='k', linewidth=0.5)
   
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax

def get_geometric_center(face):
    return (face[1]+face[3])/2

def get_qc_code(faces, vertices):
    h = np.zeros((len(faces), len(vertices)))
    for i, face in enumerate(faces):
        for j in range(len(vertices)):
            if close(face[1], vertices[j]) or close(face[2], vertices[j]) or close(face[3], vertices[j]) or close(face[4], vertices[j]):
                h[i,j] = 1
    return h

def get_classical_code_distance(h):
    if rank(h) == h.shape[1]:
        print('Code is full rank, no codewords')
        return np.inf
    else:
        start = timer()
        print('Code is not full rank, there are codewords')
        print('Computing codeword space basis ...')
        ker = nullspace(h)
        end = timer()
        print(f'Elapsed time for computing codeword space basis: {end-start} seconds', flush=True)
        print('len of ker: ', len(ker))
        print('Start finding minimum Hamming weight while buiding codeword space ...')
        start = end
        # @jit
        def find_min_weight_while_build(matrix):
            span = []
            min_hamming_weight = np.inf
            for ir, row in enumerate(matrix):
                print('debug: ir = ', ir, 'current min_hamming_weight = ', min_hamming_weight, flush=True)  # debug
                row_hamming_weight = np.sum(row)
                if row_hamming_weight < min_hamming_weight:
                    min_hamming_weight = row_hamming_weight
                temp = [row]
                for element in span:
                    newvec = (row + element) % 2
                    temp.append(newvec)
                    newvec_hamming_weight = np.sum(newvec)
                    if newvec_hamming_weight < min_hamming_weight:
                        min_hamming_weight = newvec_hamming_weight
                span = list(np.unique(temp + span, axis=0))
            assert len(span) == 2**len(matrix) - 1
            return min_hamming_weight
        min_hamming_weight = find_min_weight_while_build(ker)
        end = timer()
        print(f'Elapsed time for finding minimum Hamming weight while buiding codeword space : {end-start} seconds', flush=True)
        
        return min_hamming_weight

def get_classical_code_distance_special_treatment(h, gen, target_weight):
    if rank(h) == h.shape[1]:
        print('Code is full rank, no codewords')
        return np.inf
    else:
        start = timer()
        print('Code is not full rank, there are codewords')
        print('Computing codeword space basis ...')
        ker = nullspace(h)
        end = timer()
        print(f'Elapsed time for computing codeword space basis: {end-start} seconds', flush=True)
        print('len of ker: ', len(ker))
        print('Start finding minimum Hamming weight while buiding codeword space ...')
        start = end
        # @jit
        def find_min_weight_while_build(matrix):
            span = []
            min_hamming_weight = np.inf
            for ir, row in enumerate(matrix):
                row_hamming_weight = np.sum(row)
                if row_hamming_weight < min_hamming_weight:
                    min_hamming_weight = row_hamming_weight
                    if min_hamming_weight <= target_weight:
                        assert np.sum(row) == min_hamming_weight
                        return min_hamming_weight, row
                temp = [row]
                for element in span:
                    newvec = (row + element) % 2
                    temp.append(newvec)
                    newvec_hamming_weight = np.sum(newvec)
                    if newvec_hamming_weight < min_hamming_weight:
                        min_hamming_weight = newvec_hamming_weight
                        if min_hamming_weight <= target_weight:
                            assert np.sum(newvec) == min_hamming_weight
                            return min_hamming_weight, newvec
                span = list(np.unique(temp + span, axis=0))
            assert len(span) == 2**len(matrix) - 1
        min_hamming_weight, logical_op = find_min_weight_while_build(ker)
        end = timer()
        print(f'Elapsed time for finding minimum Hamming weight while buiding codeword space : {end-start} seconds', flush=True)
        return min_hamming_weight, logical_op


def draw_qc_code_logical(faces, vertices, h, logical_op):
    faces_pos = [np.array([get_geometric_center(face).real, get_geometric_center(face).imag]) for face in faces]
    vertices = get_vertices(faces)
    vertices_pos = [np.array([vertex.real, vertex.imag]) for vertex in vertices]
    
    fig, ax = plt.subplots()
    ax.scatter(np.array(faces_pos)[:,0], np.array(faces_pos)[:,1], marker='s', c='r')
    ax.scatter(np.array(vertices_pos)[:,0], np.array(vertices_pos)[:,1], marker='o', c='b')
    edges = get_edges(faces, vertices)
    for edge in edges:
        ax.plot([edge[0].real, edge[1].real], [edge[0].imag, edge[1].imag], color='k', linewidth=0.5)
    # for i in range(len(faces)):
        # for j in range(len(vertices)):
        #     if h[i,j] == 1:
        #         ax.plot([faces_pos[i][0], vertices_pos[j][0]], [faces_pos[i][1], vertices_pos[j][1]], color='gray', linewidth=3, zorder=-1)
    
    ones = [i for i in range(len(logical_op)) if logical_op[i] == 1]
    x = [vertices_pos[i][0] for i in ones]
    y = [vertices_pos[i][1] for i in ones]
    ax.scatter(x, y, marker='*', c='g', s=200, zorder=100)

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


gen = 1
rhombs = []

ctg = 0
A = 0.+0.j
B = 1.+0.j
C = B + np.exp(-1j*np.pi/7)
D = np.exp(-1j*np.pi/7)

# ctg = 1
# A = 0.+0.j
# B = 1.+0.j
# C = B + np.exp(-1j*np.pi/7)
# D = np.exp(-1j*np.pi/7)

# ctg = 2
# A = 0.+0.j
# B = np.exp(1j*3*np.pi/14)
# C = B -1.j
# D = 0.-1.j

# ctg = 3
# A = 0.+0.j
# B = np.exp(1j*3*np.pi/14)
# C = B -1.j
# D = 0.-1.j

# ctg = 4
# A = 0.+0.j
# B = np.exp(1j*3*np.pi/14)
# C = B -1.j
# D = 0.-1.j

# ctg = 5
# A = 0.+0.j
# B = 1.+0.j
# C = B + np.exp(-1j*3*np.pi/7)
# D = np.exp(-1j*3*np.pi/7)

# ctg = 6
# A = 0.+0.j
# B = 1.+0.j
# C = B + np.exp(-1j*3*np.pi/7)
# D = np.exp(-1j*3*np.pi/7)

rhombs.append((ctg, A, B, C, D))
for _ in range(gen):
    rhombs = subdivide(rhombs)
vertices = get_vertices(rhombs)
# fig, ax = draw_schaad(rhombs, vertices)
# plt.show()
h = get_qc_code(rhombs, vertices)
m, n = h.shape
print('m = ', m, 'n = ', n)
k = n - rank(h)
print('k = ', k)
print('d = ', get_classical_code_distance(h))


savedir = '/Users/yitan/Google Drive/My Drive/from_cannon/qmemory_simulation/data/qc_code/schaad'
subdir = f'ctg={ctg}_gen={gen}'
if not os.path.exists(os.path.join(savedir, subdir)):
    os.makedirs(os.path.join(savedir, subdir))

d_bound, logical_op = get_classical_code_distance_special_treatment(h, gen=1, target_weight=get_classical_code_distance(h))
print('d_bound = ', d_bound)
fig, ax = draw_qc_code_logical(rhombs, vertices, h, logical_op)
ax.set_title(f'low weight logical operator')
fig.set_size_inches(12, 12)


savename = f'low_weight_logical.pdf'
fig.savefig(os.path.join(savedir, subdir, savename), bbox_inches='tight')
plt.show()

# logical_basis = row_basis(nullspace(h))
# logical_coeffs = np.array(list(itertools.product([0, 1], repeat=k)))
# random sample length k binary vectors
# logical_coeffs = np.random.randint(2, size=(500, k))
# for i in range(logical_coeffs.shape[0])[:]:
#     logical_op = np.matmul(logical_coeffs[i], logical_basis) % 2
#     logical_op = logical_op.reshape((n,))
#     fig, ax = draw_qc_code_logical(rhombs, vertices, h, logical_op)
#     ax.set_title(f'logical operator {i}')
#     savename = f'logical_{i}.pdf'
#     fig.set_size_inches(12, 12)
#     fig.savefig(os.path.join(savedir, subdir, savename), bbox_inches='tight')
