# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:04:30 2018

@author: Masha

-(Laplas)u + au = f(x, y), 0 < x2 + y2 < R, 0 < y < pi
du/dn|y>0 = g(x, y), x2 + y2 = R2
u|y<0 = h(x, y) x2 + y2 = R2

Выбрать три тестовые функции и провести тестирование по алгоритму,
предложенному на лекции.
Сравнить отклонение точного аналитического решения от полученного
численно по максимум-норме и норме L2.
Сделать визуализацию получаемого численно МКЭ и аналитического
решений с помощью библиотеки matplotlib
"""
import numpy
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from fenics import *
from mshr import *

R = pi
er = 1E-16
domain = Circle(Point(0, 0), R)
mesh = generate_mesh(domain, 32)

V = FunctionSpace(mesh, 'P', 1)

# u = h
def boundary(x, on_boundary):
    return on_boundary or (x[1] < 0)

u_D = Expression('3*x[1]*sin(x[0])', degree = 2)

alpha = 2
f = Expression('(-1 + alpha) * 3*x[1]*sin(x[0])', degree = 2, alpha = alpha)
g = Expression('-3*x[1]*cos(x[0])*x[0]/R - 3*sin(x[0])*x[1]/R', degree = 2, R = R)
#h = Expression('3*x[1]*sin(x[0])', degree = 2)

u = TrialFunction(V)
v = TestFunction(V)

a = dot(grad(u), grad(v))*dx + alpha*u*v*ds
L = f*v*dx - g*v*ds

bc = DirichletBC(V, u_D, boundary)
u = Function(V)
solve(a == L, u, bc)

#errors
error_L2 = errornorm(u_D, u, 'L2')
print('error_L2 =', error_L2)

vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_C = numpy.max(numpy.abs(vertex_values_u - vertex_values_u_D))
print('error_C =', error_C)

#visualisation
n = mesh.num_vertices()
d = mesh.geometry().dim()
mesh_coordinates = mesh.coordinates().reshape((n, d))
triangles = numpy.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
plt.figure()
zfaces = numpy.asarray([u_D(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
plt.title("Analytical solution")
plt.savefig('kz_1a.png')

plt.figure()
zfaces = numpy.asarray([u(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
plt.title("Numerical solution")
plt.savefig('kz_1n.png')
