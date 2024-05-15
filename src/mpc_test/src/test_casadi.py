import casadi

opti = casadi.Opti()
x,y = opti.variable(), opti.variable()

opti.minimize((y-x**2)**2)
opti.subject_to(x**2+y**2==1)
opti.subject_to(x+y>=1)
opti.solver('ipopt')
sol = opti.solve()

print(sol)