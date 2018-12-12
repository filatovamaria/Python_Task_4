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
import math
       

from fenics import *
from mshr import *

T = 1
steps = 10
dt = T / steps
filenames = []
R = 1
er = 1E-16
domain = Circle(Point(0, 0), R)
mesh = generate_mesh(domain, 64)
V = FunctionSpace(mesh, 'P', 2)

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

alpha = 2
beta = 3
t = 0

# gamma = 1.2 #main formula
#u_D = Expression('1+x[0]*x[0]+alpha*x[1]*x[1]+beta*t',degree = 2, alpha = alpha, beta = beta, t = 0)
#
#f = Expression('beta - 2*gamma*(1+alpha)', degree=2, alpha = alpha, gamma = gamma, beta = beta)
#g = Expression('2*x[0]*x[0]/R + 2*alpha*x[1]*x[1]/R', degree=2, R = R, alpha = alpha, beta = beta)


#gamma = 2 #main formula
#u_D = Expression('3*x[1]*sin(x[0])*t', degree = 2, t = t)
#
#f = Expression('(1 + gamma*t) * 3*x[1]*sin(x[0])', degree = 2, gamma = gamma, t = t)
#g = Expression('-3*x[1]*cos(x[0])*x[0]*t/R - 3*sin(x[0])*x[1]*t/R', degree = 2, R = R, t = t)

gamma = 10 #main formula
u_D = Expression('-x[1] * x[0] * t', degree = 2, t = t) # == h

f = Expression('- gamma * x[1] * x[0]', degree = 2, gamma = gamma)
g = Expression('x[1]*x[0]*t/R + x[0]*x[1]*t/R', degree = 2, R = R, t = t)

bc = DirichletBC(V, u_D, bounds, 1)
u_i = interpolate(u_D,V)

u = TrialFunction(V)
v = TestFunction(V)

a = gamma*dt * dot(grad(u),grad(v))*dx + u*v*dx
L = (u_i + dt*f)*v*dx + gamma * dt * g * v *ds(0, subdomain_data = bounds)


n = mesh.num_vertices()
d = mesh.geometry().dim()
mesh_coordinates = mesh.coordinates().reshape((n,d))
triangles = numpy.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:,0],mesh_coordinates[:,1],triangles)

errL2 = []
errC = []
tt = []
u = Function(V)

for n in range(steps):
    
    t += dt
    tt.append(t)
    u_D.t = t

    solve(a==L, u, bc)

    u_e = interpolate(u_D,V)
    
    vertex_values_u_e = u_e.compute_vertex_values(mesh)
    vertex_values_u   = u.compute_vertex_values(mesh)

    error_L2 = errornorm(u_e, u, 'L2')
    error_C = numpy.max(numpy.abs(vertex_values_u - vertex_values_u_e))
    
    print('t = ', t, ', error_L2 = ', error_L2,'\n', '        error_C = ', error_C)
    errL2.append(error_L2)
    errC.append(error_C)
    
    plt.figure()    
    zfaces = numpy.asarray([u(cell.midpoint()) for cell in cells(mesh)])
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    #plt.clim(4.1,5.2)
    #plt.clim(-1.5, 1.5)
    plt.clim(-0.5, 0.5)
    plt.colorbar()
    plt.title(('Sol: t = '+"{:.1f}".format(t)+', error_L2 = '+"{:.5f}".format(error_L2)+', error_C = '+ "{:.5f}".format(error_C)))
    plt.savefig('heat_s_'+"{:.1f}".format(t)+'.png')
    
    plt.figure()
    zfaces = numpy.asarray([u_e(cell.midpoint()) for cell in cells(mesh)])
    plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
    plt.title(('An: t = '+"{:.1f}".format(t)+', error_L2 = '+"{:.5f}".format(error_L2)+', error_C = '+ "{:.5f}".format(error_C)))
    #plt.clim(4.1,5.2)
    #plt.clim(-1.5, 1.5)
    plt.clim(-0.5, 0.5)
    plt.colorbar()
    plt.savefig('heat_a_'+"{:.1f}".format(t)+'.png')
    u_i.assign(u)
    
plt.figure()
plt.title("Errors")
plt.plot(tt, errL2, 'r-', label= 'L2')
plt.plot(tt, errC, 'g-', label = 'C')
plt.legend()
plt.grid()
plt.savefig('errors.png')
