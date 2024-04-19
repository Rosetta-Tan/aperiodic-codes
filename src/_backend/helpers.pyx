def subdivide(triangles: List[Tuple[int, complex, complex, complex]]) -> List[Tuple[int, complex, complex, complex]]:
    result: List[Tuple[int, complex, complex, complex]] = []
    for ctg, A, B, C in triangles:
        if ctg == 0:
            P1 = A + 2/5*(B-A)
            P2 = A + 4/5*(B-A)
            P3 = 0.5*(A+C)
            P4 = 0.5*(P2+C)

            F1 = (0, A, P3, P1)
            F2 = (0, P2, P3, P1)
            F3 = (0, P3, P2, P4)
            F4 = (0, P3, C, P4)
            F5 = (0, C, B, P2)
            result += [F1, F2, F3, F4, F5] 
    return result
