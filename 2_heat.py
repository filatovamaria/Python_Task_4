# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 22:09:55 2018

@author: Masha

du/dt = a(Laplas)u + f(x, y, t), 0 < x2 + y2 < R, 0 < y < pi
du/dn|y>0 = g(x, y, t), x2 + y2 = R2
u|y<0 = h(x, y, t) x2 + y2 = R2

Выбрать три тестовые функции и провести тестирование по алгоритму,
предложенному на лекции.
Сравнить отклонение точного аналитического решения от полученного
численно по максимум-норме и норме L2.
Сделать визуализацию получаемого численно МКЭ и аналитического
решений с помощью библиотеки matplotlib, результат расчета временной
задачи сохранить в avi или gif.
"""
import numpy
import matplotlib.pyplot as plt
import matplotlib.tri as tri
       

from fenics import *
from mshr import *

T = 1
steps = 10
dt = 0.1 #T / steps
filenames = []
R = 1
er = 1E-16
domain = Circle(Point(0, 0), R)
mesh = generate_mesh(domain, 64)
V = FunctionSpace(mesh, 'P', 2)

alpha = 1
beta = 1.3
gamma = 1.2

def boundary(x, on_boundary):
    return on_boundary or (x[1] < 0)

u_D = Expression('1+x[0]*x[0]+gamma*x[1]*x[1]+beta*t',degree = 2, gamma = gamma, beta = beta, t = 0)

f = Expression('beta - 2*alpha*(1+gamma)', degree=2, alpha = alpha, gamma = gamma, beta = beta)
g = Expression('-2*x[0]*x[0]/R - 2*gamma*x[1]*x[1]/R', degree=2, R = R, gamma = gamma, beta = beta)

bc = DirichletBC(V, u_D, boundary)
u_i = interpolate(u_D,V)

u = TrialFunction(V)
v = TestFunction(V)

a = alpha*dt * dot(grad(u),grad(v))*dx + u*v*dx
L = (u_i + dt*f)*v*dx - alpha * dt * g * v *ds

t = 0

n = mesh.num_vertices()
d = mesh.geometry().dim()
mesh_coordinates = mesh.coordinates().reshape((n,d))
triangles = numpy.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:,0],mesh_coordinates[:,1],triangles)

for n in range(steps):
    
    t += dt
    u_D.t = t
    u = Function(V)
    solve(a==L, u, bc)

    zfaces = numpy.asarray([u(cell.midpoint()) for cell in cells(mesh)])
    u_e = interpolate(u_D,V)
    
    vertex_values_u_e = u_e.compute_vertex_values(mesh)
    vertex_values_u   = u.compute_vertex_values(mesh)

    error_L2 = errornorm(u_e, u, 'L2')
    error_C = numpy.max(numpy.abs(vertex_values_u - vertex_values_u_e))
    
    print('t=', t, ', error_L2 = ', error_L2,'\n', '        error_C = ', error_C)
    plt.figure()
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    plt.title('Sol: t = '+ str(t) + ', error_L2 = '+ str(error_L2) + ', error_C = ' + str(error_C))
    plt.savefig('heat_s_'+ str(t) +'.png')
    
    zfaces = numpy.asarray([u_e(cell.midpoint()) for cell in cells(mesh)])
    plt.figure()
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    plt.title('An: t = '+ str(t) + ', error_L2 = '+ str(error_L2) + ', error_C = ' + str(error_C))
    plt.savefig('heat_a_'+ str(t) +'.png')
    u_i.assign(u)