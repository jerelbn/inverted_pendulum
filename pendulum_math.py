import sympy as sp

sp.init_printing()

ddx = sp.Symbol('ddx')
ddt = sp.Symbol('ddtheta')
dx = sp.Symbol('dx')
dt = sp.Symbol('dtheta')
x = sp.Symbol('x')
t = sp.Symbol('theta')
M = sp.Symbol('M')
m = sp.Symbol('m')
J = sp.Symbol('J')
g = sp.Symbol('g')
l = sp.Symbol('l')
b = sp.Symbol('b')
c = sp.Symbol('c')
F = sp.Symbol('F')

# Equations of motion
eq1 = (M + m) * ddx + b * dx + m * l * ddt * sp.cos(t) - m * l * dt ** 2 * sp.sin(t) - F
eq2 = (J + m * l ** 2) * ddt + m * ddx * l * sp.cos(t) + m * g * l * sp.sin(t) + c * dt

# Solve for ddx and ddt
sol = sp.solve([eq1, eq2], (ddx, ddt))
eq_ddx = sp.simplify(sol[ddx])
eq_ddt = sp.simplify(sol[ddt])
print(f'ddx = {eq_ddx}')
print(f'ddtheta = {eq_ddt}')

# LQR matrices A, B
# state: [x, dx, theta, dtheta]
# dstate: [dx, ddx, dtheta, ddtheta]

# Linearize about theta = pi
#   sin(theta) ~ -theta
#   cos(theta) ~ -1
#   squared terms ~ 0
eq1_1 = (M + m) * ddx + b * dx - m * l * ddt - F
eq2_1 = (J + m * l ** 2) * ddt - m * ddx * l - m * g * l * t + c * dt
sol = sp.solve([eq1_1, eq2_1], (ddx, ddt))
eq_dx = dx
eq_dt = dt
eq_ddx = sp.simplify(sol[ddx])
eq_ddt = sp.simplify(sol[ddt])

# A
d_dx_dx = sp.simplify(sp.diff(eq_dx, x))
d_dx_ddx = sp.simplify(sp.diff(eq_dx, dx))
d_dx_dt = sp.simplify(sp.diff(eq_dx, t))
d_dx_ddt = sp.simplify(sp.diff(eq_dx, dt))

d_ddx_dx = sp.simplify(sp.diff(eq_ddx, x))
d_ddx_ddx = sp.simplify(sp.diff(eq_ddx, dx))
d_ddx_dt = sp.simplify(sp.diff(eq_ddx, t))
d_ddx_ddt = sp.simplify(sp.diff(eq_ddx, dt))

d_dt_dx = sp.simplify(sp.diff(eq_dt, x))
d_dt_ddx = sp.simplify(sp.diff(eq_dt, dx))
d_dt_dt = sp.simplify(sp.diff(eq_dt, t))
d_dt_ddt = sp.simplify(sp.diff(eq_dt, dt))

d_ddt_dx = sp.simplify(sp.diff(eq_ddt, x))
d_ddt_ddx = sp.simplify(sp.diff(eq_ddt, dx))
d_ddt_dt = sp.simplify(sp.diff(eq_ddt, t))
d_ddt_ddt = sp.simplify(sp.diff(eq_ddt, dt))

print(f'A = array([[{d_dx_dx}, {d_dx_ddx}, {d_dx_dt}, {d_dx_ddt}],')
print(f'           [{d_ddx_dx}, {d_ddx_ddx}, {d_ddx_dt}, {d_ddx_ddt}],')
print(f'           [{d_dt_dx}, {d_dt_ddx}, {d_dt_dt}, {d_dt_ddt}],')
print(f'           [{d_ddt_dx}, {d_ddt_ddx}, {d_ddt_dt}, {d_ddt_ddt}]])')

# B
d_dx_F = sp.simplify(sp.diff(eq_dx, F))
d_ddx_F = sp.simplify(sp.diff(eq_ddx, F))
d_dt_F = sp.simplify(sp.diff(eq_dt, F))
d_ddt_F = sp.simplify(sp.diff(eq_ddt, F))

print(f'B = array([[{d_dx_F}],')
print(f'           [{d_ddx_F}],')
print(f'           [{d_dt_F}],')
print(f'           [{d_ddt_F}]])')
