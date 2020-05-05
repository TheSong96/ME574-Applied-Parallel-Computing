import numpy as np
##Import other packages here##

pi_string = "3141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233786783165271201909145648566923460348610454326648213393607260249141273724587006606315588174881520920962829254091715364367892590360011330530548820466521384146951941511609433057270365759591953092186117381932611793105118548074462379962749567351885752724891227938183011949129833673362440656643086021394946395224737190702179860943702770539217176293176752384674818467669405132000568127145263560827785771342757789609"
pi_digits = [int(char) for char in pi_string]
v = 0.1*np.array(pi_digits)[0:374]

def ewprod(u,v):
    '''
    Compute the element-wise product of arrays
	Must call GPU kernel function
    Args:
        u,v: 1D numpy arrays

    Returns:
        w: 1D numpy array containing product of corresponding entries in input arrays u and v
    '''

    return w

def smooth_serial(v,rad):
    '''
    compute array of local averages (with radius rad)
    '''
    return w

def smooth_parallel(v,rad):
    '''
    compute array of local averages (with radius rad)
	Must call GPU kernel function
    '''

    return d_out.copy_to_host()

def smooth_parallel_sm(v,rad):
    '''
    compute array of local averages (with radius rad)
	Must call GPU kernel function & utilize shared memory
    '''

    return d_out.copy_to_host()

def rk4_step(f,y,t0,h):
    """
    compute next value for 4th order Runge Kutta ODE solver
    
    Args:
        f: right-hand side function that gives rate of change of y
        y: float value of dependent variable
        t0: float initial value of independent variable
        h: float time step
        
    Returns:
        y_new: float estimated value of y(t0+h)
    """
    
    return y_new

def rk_solve(f,y0,t):
    """
    Runge-Kutta solver for systems of 1st order ODEs
    (should call function rk4_step)
    
    Args:
        f: name of right-hand side function that gives rate of change of y
        y0: numpy array of initial float values of dependent variable
        t: numpy array of float values of independent variable
        
    Returns:
        y: 2D numpy array of float values of dependent variable
    """
    
    return y

@cuda.jit(device = True)
def rk4_step_parallel(f,y,t0,h,w):
    """
    compute next value for 4th order Runge Kutta ODE solver
    
    Args:
        f: right-hand side function that gives rate of change of y
        y: float value of dependent variable
        t0: float initial value of independent variable
        h: float time step
        w: float of omega value
    """
    pass

@cuda.jit
def rk4_solve_parallel(f,y,t,w):
    """
    Runge-Kutta solver for systems of 1st order ODEs
    (should call function rk4_step_parallel)
    
    Args:
        f: name of right-hand side function that gives rate of change of y
        y: numpy array of dependent variable output (fist entry should be initial conditions)
        t: numpy array of float values of independent variable
        w: numpy array of omega values
    """
    pass

#OPTIONAL#
def eigenvals_parallel(NC,NW,NT,h):
	"""
	Determines eigenvalues of y''+w^2*y = 0, y'(pi) -c*y(pi) = 0 for w : [0,10] and c : [0,5]
	y(0) = 0, y'(0) = 1
	(should call a 2D gpu kernel)

	Args:
        NC: Number of samples in the range [0,5] to take for c
        NW: Number of samples in the range [0,10] to take for w
        NT: Number of timesteps to run each instance of the GPU-based rk4 solver
        h: step size for rk4 solver
        
    Returns:
        eigs: 2D numpy array of shape NCxNW containing eigenvalues as a function of c and w
	"""

	#return eigs
	pass
##########

def jacobi_update_serial(u,f):
    """
    Performs jacobi iteration update for 2D Poisson equation
    Boundary conditions: u(0,0)=1; outside boundary defined by f, u(x<0)=-1, u(x>0)=0.5
    
    Args:
    	x: vector specifying x coordinates
    	y: vector specifying y coordinates
        u: 2D input array for update (u.shape = (len(x),len(y)))
        f: function specifying boundary of region to evaluation Poission equation

    Precondtion:
    	The cordinate system defined by vectors x and y contains the point (0,0) and spans 
    	the entire boundary defined by f
        
    Returns:
        v: updated version of u
    """
    return v

@cuda.jit
def jacobi_update_kernel(d_u,d_x,d_y):
    """
    Performs jacobi iteration update for 2D Poisson equation
    Boundary conditions: u(0,0)=1; outside boundary defined by f, u(x<0)=-1, u(x>0)=0.5
    where f = x^4 + y^4 - 1
    
    Args:
    	d_x: vector specifying x coordinates
    	d_y: vector specifying y coordinates
        d_u: 2D input array for update (u.shape = (len(x),len(y)))

    Precondtion:
    	The cordinate system defined by vectors x and y contains the point (0,0) and spans 
    	the entire boundary defined by f
    """
    pass

if __name__ == "__main__":

	#Problem 1
	#(should call function ewprod)

	#Problem 2
	# a) (should call function smooth_serial)

	# b) (should call function smooth_parallel)

	# c) (should call function smooth_parallel_sm)

	#Problem 3
	# a) (should call function rk4_solve)

	# b/c) (should call function rk4_solve_parallel)

	# d) OPTIONAL (should call function eigenvals_parallel)

	#Problem 4
	# (should call function jacobi_update_serial)

	#Problem 5
	# (should call function jacobi_update_kernel)