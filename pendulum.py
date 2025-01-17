import time
import pygame as pg
from numpy import pi, sin, cos, array, sign, diag, random
from numpy.linalg import inv
from scipy.linalg import solve_continuous_are

random.seed(int(time.time()))

SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 500
FPS = 60
SCREEN_CENTER_X = SCREEN_WIDTH/2
SCREEN_CENTER_Y = SCREEN_HEIGHT/2
PIXELS_PER_METER = 100
CENTER_DRAWING = False

GRAVITY = 9.81
MASS_CART = 1.0
MASS_PENDULUM = 0.1
LINEAR_DRAG_COEFF = 0.1
ANGULAR_DRAG_COEFF = 0.001
CART_WIDTH = 0.2
CART_HEIGHT = 0.1
PENDULUM_LENGTH = 1.0
PENDULUM_WIDTH = 0.05
INERTIA = MASS_PENDULUM * PENDULUM_LENGTH**2

X_INIT = SCREEN_CENTER_X / PIXELS_PER_METER + random.uniform(-1.0, 1.0)
DX_INIT = random.uniform(-0.1, 0.1)
THETA_INIT = random.uniform(-pi/6, pi/6)
DTHETA_INIT = random.uniform(-0.1, 0.1)

MAX_FORCE = 50.0
CONTROL_SWITCH_ANGLE = 30.0 * pi / 180

LQR_ERRX_MAX = 5.0
LQR_RELINEARIZE_ERROR = 5.0 * pi / 180
LQR_Q = diag([10, 1, 100, 1])
LQR_R = array([[1]])

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
CYAN = (0,255,255)

# Wrap between 0 and 2*pi
def wrap_angle(angle):
  while angle < 0:
    angle += 2*pi
  while angle >= 2*pi:
    angle -= 2*pi
  return angle

# Angle difference accounting for wrap-around
def angle_diff(a, b):
  diff = a - b
  while diff > pi:
    diff -= 2*pi
  while diff < -pi:
    diff += 2*pi
  return diff

# 4th order Runge-Kutta method
def rk4(func, state, input, step):
  k1 = func(state, input)
  k2 = func(state + step/2*k1, input)
  k3 = func(state + step/2*k2, input)
  k4 = func(state + step*k3, input)
  return state + step/6*(k1 + 2*k2 + 2*k3 + k4)
  
# State: [x, dx, theta, dtheta]
def dynamics(state, input):
  g = GRAVITY
  M = MASS_CART
  m = MASS_PENDULUM
  J = INERTIA
  b = LINEAR_DRAG_COEFF
  c = ANGULAR_DRAG_COEFF
  l = PENDULUM_LENGTH
  F = input[0]
  dx = state[1]
  theta = state[2]
  dtheta = state[3]
  D = J*M + J*m + M*l**2*m + l**2*m**2*sin(theta)**2
  ddx = (F*J + F*l**2*m - J*b*dx + J*dtheta**2*l*m*sin(theta) - b*dx*l**2*m + c*dtheta*l*m*cos(theta) + dtheta**2*l**3*m**2*sin(theta) + g*l**2*m**2*sin(2*theta)/2)/D
  ddtheta = (-F*l*m*cos(theta) - M*c*dtheta - M*g*l*m*sin(theta) + b*dx*l*m*cos(theta) - c*dtheta*m - dtheta**2*l**2*m**2*sin(2*theta)/2 - g*l*m**2*sin(theta))/D
  return array([dx, ddx, dtheta, ddtheta])

def compute_control_lqr(state, state_ref, F_prev):
  if not hasattr(compute_control_lqr, 'K'):
    compute_control_lqr.K = None

  g = GRAVITY
  M = MASS_CART
  m = MASS_PENDULUM
  J = INERTIA
  b = LINEAR_DRAG_COEFF
  c = ANGULAR_DRAG_COEFF
  l = PENDULUM_LENGTH

  x = state[0]
  dx = state[1]
  theta = state[2]
  dtheta = state[3]

  x_ref = state_ref[0]
  dx_ref = state_ref[1]
  theta_ref = state_ref[2]
  dtheta_ref = state_ref[3]

  err_x = max(min(x_ref - x, LQR_ERRX_MAX), -LQR_ERRX_MAX)
  err_dx = dx_ref - dx
  err_theta = angle_diff(theta_ref, theta)
  err_dtheta = dtheta_ref - dtheta

  if err_theta > LQR_RELINEARIZE_ERROR or compute_control_lqr.K is None:
    F = F_prev
    D = J*M + J*m + M*l**2*m + l**2*m**2*sin(theta)**2
    A = array([[0, 1, 0, 0],
            [0, b*(-J - l**2*m)/D, l*m*(-l*m*(2*F*J + 2*F*l**2*m - 2*J*b*dx + 2*J*dtheta**2*l*m*sin(theta) - 2*b*dx*l**2*m + 2*c*dtheta*l*m*cos(theta) + 2*dtheta**2*l**3*m**2*sin(theta) + g*l**2*m**2*sin(2*theta))*sin(theta)*cos(theta) + D*(J*dtheta**2*cos(theta) - c*dtheta*sin(theta) + dtheta**2*l**2*m*cos(theta) + g*l*m*cos(2*theta)))/D**2, l*m*(2*J*dtheta*sin(theta) + c*cos(theta) + 2*dtheta*l**2*m*sin(theta))/D],
            [0, 0, 0, 1],
            [0, b*l*m*cos(theta)/D, l*m*(l*m*(2*F*l*m*cos(theta) + 2*M*c*dtheta + 2*M*g*l*m*sin(theta) - 2*b*dx*l*m*cos(theta) + 2*c*dtheta*m + dtheta**2*l**2*m**2*sin(2*theta) + 2*g*l*m**2*sin(theta))*sin(theta)*cos(theta) + D*(F*sin(theta) - M*g*cos(theta) - b*dx*sin(theta) - dtheta**2*l*m*cos(2*theta) - g*m*cos(theta)))/D**2, -(M*c + c*m + dtheta*l**2*m**2*sin(2*theta))/D]])
    B = array([[0],
            [(J + l**2*m)/D],
            [0],
            [-l*m*cos(theta)/D]])
    X = solve_continuous_are(A, B, LQR_Q, LQR_R)
    K = inv(LQR_R) @ B.T @ X

  F = (K @ array([err_x, err_dx, err_theta, err_dtheta]))[0]
  # print('state: [%9.4f, %9.4f, %9.4f, %9.4f], err_x: %9.4f, err_theta: %9.4f, F: %9.4f' % (state[0], state[1], state[2], state[3], err_x, err_theta, F))
  return F

def compute_control_energy(state, state_ref):
  g = GRAVITY
  m = MASS_PENDULUM
  J = INERTIA
  l = PENDULUM_LENGTH

  theta = state[2]
  dtheta = state[3]

  E_ref = 2.0 * m * g * l
  E = 0.5 * (J + m * l ** 2) * dtheta ** 2 + m * g * l * (1 - cos(theta))
  E_tilde = E - E_ref

  k = 100
  kx = 10

  F_energy = k * E_tilde * sign(dtheta * cos(theta) + 1e-9)
  F_x = kx * (state_ref[0] - state[0])
  F = F_energy + F_x

  # print('F_energy: %9.4f, F_x: %9.4f' % (F_energy, F_x))
  # print('state: [%9.4f, %9.4f, %9.4f, %9.4f], E: %9.4f, F: %9.4f' % (state[0], state[1], state[2], state[3], E, F))

  return F

# Inputs in pixels
def draw_pendulum(screen, x, y, theta, x_ref):
  # Center cart on screen
  if CENTER_DRAWING:
    x = SCREEN_CENTER_X

  # Cart
  cw = int(PIXELS_PER_METER * CART_WIDTH)
  ch = int(PIXELS_PER_METER * CART_HEIGHT)
  pg.draw.rect(screen, RED, rect=(x-cw/2, y-ch/2, cw, ch), width=0)

  # Pendulum
  pl = int(PIXELS_PER_METER * PENDULUM_LENGTH)
  pw = int(PIXELS_PER_METER * PENDULUM_WIDTH)
  xp = x + pl * sin(theta)
  yp = y + pl * cos(theta)
  pg.draw.line(screen, GREEN, start_pos=(x, y), end_pos=(xp, yp), width=pw)

  # Control point
  pg.draw.line(screen, CYAN, start_pos=(int(x), SCREEN_CENTER_Y), end_pos=(int(x_ref), SCREEN_CENTER_Y), width=1)
  pg.draw.circle(screen, CYAN, (int(x_ref), SCREEN_CENTER_Y), 3)

def main():
  pg.init()
  pg.display.set_caption('Inverted Pendulum')
  screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
  clock = pg.time.Clock()

  state = array([X_INIT, DX_INIT, THETA_INIT, DTHETA_INIT])
  state_ref = array([SCREEN_CENTER_X/PIXELS_PER_METER, 0, pi, 0])
  input = array([0.0])

  running = True
  while running:
    # Compute control input
    state[2] = wrap_angle(state[2])
    if abs(state[2] - pi) < CONTROL_SWITCH_ANGLE:
      input[0] = compute_control_lqr(state, state_ref, input[0])

      # Adjust horizontal set point based on mouse position
      mp = pg.mouse.get_pos()
      state_ref[0] = mp[0] / PIXELS_PER_METER
    else:
      input[0] = compute_control_energy(state, state_ref)
    input[0] = max(min(input[0], MAX_FORCE), -MAX_FORCE)

    # Integrate dynamics
    dt = 1.0 / FPS
    state = rk4(dynamics, state, input, dt)

    # Drawing
    screen.fill(BLACK)
    draw_pendulum(screen, state[0]*PIXELS_PER_METER, SCREEN_CENTER_Y, state[2], state_ref[0]*PIXELS_PER_METER)
    pg.display.flip()

    # Handle key presses
    keys = pg.key.get_pressed()
    for event in pg.event.get():
      if event.type == pg.QUIT:
        running = False
    if keys[pg.K_ESCAPE]:
      running = False
    if keys[pg.K_LEFT]:
      state[2] -= 5.0 * pi / 180
    if keys[pg.K_RIGHT]:
      state[2] += 5.0 * pi / 180

    clock.tick(FPS)

  pg.quit()

if __name__ == '__main__':
  main()
