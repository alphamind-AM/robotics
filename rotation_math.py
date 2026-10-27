import numpy as np
import sympy as sp

#symbols
theta = sp.symbols('theta')
tx, ty, tz = sp.symbols('tx ty tz')

def rotation_matrix(theta, axis):
    if axis == 'x':
        return sp.Matrix([[1, 0, 0, 0],
                            [0, sp.cos(theta), -sp.sin(theta), 0],
                            [0, sp.sin(theta), sp.cos(theta), 0],
                            [0, 0, 0, 1]])
    elif axis == 'y':
        return sp.Matrix([[sp.cos(theta), 0, sp.sin(theta), 0],
                            [0, 1, 0, 0],
                            [-sp.sin(theta), 0, sp.cos(theta), 0],
                            [0, 0, 0, 1]])
    elif axis == 'z':
        return sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0, 0],
                            [sp.sin(theta), sp.cos(theta), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")


def translation_matrix(tx, ty, tz):
    return sp.Matrix([[1, 0, 0, tx],
                      [0, 1, 0, ty],
                      [0, 0, 1, tz],
                      [0, 0, 0, 1]])



def dan_hart(theta, offset, lenght, twist):
    R_z = rotation_matrix(theta, "z")
    R_x = rotation_matrix(twist, "x")
    T_z = translation_matrix(0, 0, offset)
    T_x = translation_matrix(lenght, 0, 0)
    A = R_z * T_z * T_x * R_x 
    return A

def euler_rotation(phi, theta, psi):
    R_x = rotation_matrix(phi, "x")
    R_y = rotation_matrix(theta, 'y')
    R_z = rotation_matrix(psi, 'z')
    R_zyx = R_z * R_y * R_x
    return R_zyx

def extract_eular_angles(T_matrix):
    R_11 = T_matrix[0,0]
    R_21 = T_matrix[1,0]
    R_31 = T_matrix[2,0]
    R_32 = T_matrix[2,1]
    R_33 = T_matrix[2,2]
    phi = sp.atan2(R_21, R_11)
    theta = sp.atan2(-R_31, sp.sqrt(R_11**2+R_21**2))
    psi = sp.atan2(R_32, R_33)
    return phi, theta, psi



if False:
    d_i = sp.symbols("d_i")
    a_i = sp.symbols("a_i")
    alpha = sp.symbols('alpha')
    A_I_example = dan_hart(theta,d_i,a_i,alpha)
    sp.pprint(A_I_example)



theta_1 = sp.symbols('theta_1')
theta_2 = sp.symbols('theta_2')
theta_3 = sp.symbols('theta_3')
theta_4 = sp.symbols('theta_4')
theta_5 = sp.symbols('theta_5')

d_1 = sp.symbols("d_1")

if True:
    theta_1 = sp.rad(0)
    theta_2 = sp.rad(0)
    theta_3 = sp.rad(0)
    theta_4 = sp.rad(0)
    theta_5 = sp.rad(0)
    d_1 = 0

A_1 = dan_hart(theta_1, 0, 0, sp.rad(-90))
A_2 = dan_hart(theta_2, 0.1, 0, sp.rad(90))
A_3 = dan_hart(0, d_1, 0,0)
A_4 = dan_hart(theta_3, 0, 0, sp.rad(-90))
A_5 = dan_hart(theta_4, 0, 0, sp.rad(90))
A_6 = dan_hart(theta_5, 0, 0, 0)
out = A_1 * A_2 * A_3 * A_4 * A_5 * A_6
#sp.pprint(out)


e = euler_rotation(sp.symbols('phi'), sp.symbols('theta'), sp.symbols('psi'))
# sp.pprint(e)
sp.pprint(extract_eular_angles(out))
