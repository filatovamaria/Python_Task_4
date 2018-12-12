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

domain = Circle(Point(0, 0), R)
mesh = generate_mesh(domain, 64)

V = FunctionSpace(mesh, 'P', 1)

bounds = MeshFunction("size_t", mesh, 1)

# du/dn|y>0 = g(x,y)
class boundary1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary or (x[1] >= 0)
    
b1 = boundary1()
b1.mark(bounds, 0)

# u|(y<0) = h(x,y), on boundary and y<0
class boundary2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary or (x[1] < 0)

b2 = boundary2()
b2.mark(bounds, 1)

#u_D = Expression('3*x[1]*sin(x[0])', degree = 2)
#
#alpha = 2
#f = Expression('(1 + alpha) * 3*x[1]*sin(x[0])', degree = 2, alpha = alpha)
#g = Expression('-3*x[1]*cos(x[0])*x[0]/R - 3*sin(x[0])*x[1]/R', degree = 2, R = R)

u_D = Expression('5*x[1]*exp(x[0]) - 2', degree = 2) # == h

alpha = -1
f = Expression('(alpha - 1) * 5 * x[1] * exp(x[0]) - 2 * alpha', degree = 2, alpha = alpha)
g = Expression('-5*x[1]*exp(x[0])*x[0]/R - 5*exp(x[0])*x[1]/R', degree = 2, R = R)

#u_D = Expression('-x[1]*x[0]', degree = 2) # == h
#
#alpha = 10
#f = Expression('- alpha * x[1] * x[0]', degree = 2, alpha = alpha)
#g = Expression('x[1]*x[0]/R + x[0]*x[1]/R', degree = 2, R = R)

u = TrialFunction(V)
v = TestFunction(V)
bc = DirichletBC(V, u_D, bounds, 1)

a = dot(grad(u), grad(v))*dx + alpha*dot(u,v)*dx
L = f*v*dx - g*v*ds(0, subdomain_data = bounds)

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
plt.savefig('kz_1a1.png')

plt.figure()
zfaces = numpy.asarray([u(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
plt.title("Numerical solution")
plt.savefig('kz_1n1.png')
