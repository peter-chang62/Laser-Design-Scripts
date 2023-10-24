from sympy.parsing.mathematica import parse_mathematica

file = open("parse_mathematica.txt", "r")
n1 = parse_mathematica(file.readline().split("\n")[0])
n2 = parse_mathematica(file.readline().split("\n")[0])
n3 = parse_mathematica(file.readline().split("\n")[0])
n4 = parse_mathematica(file.readline().split("\n")[0])
n5 = parse_mathematica(file.readline().split("\n")[0])
file.close()
