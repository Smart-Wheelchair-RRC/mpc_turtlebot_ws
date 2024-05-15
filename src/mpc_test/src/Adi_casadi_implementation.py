import casadi as ca
import numpy as np
from casadi import MX, vertcat, sumsqr, norm_2
from casadi import SX, vertcat, norm_2, nlpsol

class DifferentialDriveRobot:
    def __init__(self, robot_id, initial_state, goal_state, N, max_linear_velocity, max_angular_velocity, safety_distance, obstacles=[]):
        self.id = robot_id
        self.initial_state = np.array(initial_state)  # [x, y, theta]
        self.goal_state = np.array(goal_state)        # [x, y, theta]
        self.N = N  # Planning horizon
        self.obstacles = obstacles  # Dynamic obstacles (humans)
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.safety_distance = safety_distance

        # State variables (x, y, theta)
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.theta = ca.SX.sym('theta')
        self.states = ca.vertcat(self.x, self.y, self.theta)
        self.n_states = self.states.numel()

        # Control variables (linear and angular velocities)
        self.v = ca.SX.sym('v')
        self.omega = ca.SX.sym('omega')
        self.controls = ca.vertcat(self.v, self.omega)
        self.n_controls = self.controls.numel()

        # Setup the MPC problem
        self.setup_mpc_problem()

    def setup_mpc_problem(self):
        # Define the state transition function for differential drive
        dt = 0.1  # time step
        self.state_transition = ca.vertcat(
            self.x + self.v * ca.cos(self.theta) * dt,
            self.y + self.v * ca.sin(self.theta) * dt,
            self.theta + self.omega * dt
        )

        # Cost function parameters
        self.q1 = 1  # Weight for the distance to the goal
        self.q2 = 1  # Weight for the smoothness of the trajectory
        self.q3 = 10  # Weight for obstacle avoidance

        # Constraints setup
        self.v_min = -self.max_linear_velocity
        self.v_max = self.max_linear_velocity
        self.omega_min = -self.max_angular_velocity
        self.omega_max = self.max_angular_velocity

        # Define optimization variables
        self.X = MX.sym('X', self.n_states, self.N+1)  # States over the horizon
        self.U = MX.sym('U', self.n_controls, self.N)  # Control inputs over the horizon

        # Cost function
        self.cost = 0
        self.g = []  # constraints vector
        self.lbg = []  # lower bound for g
        self.ubg = []  # upper bound for g

        for k in range(self.N):
            # Cost related to the goal
            self.cost += self.q1 * sumsqr(self.X[:2, k] - self.goal_state[:2])
            self.cost += self.q2 * sumsqr(self.X[2, k] - self.goal_state[2])
            
            # Cost related to smoothness
            if k > 0:
                self.cost += self.q3 * sumsqr(self.U[:, k] - self.U[:, k-1])

            # State transition
            st = self.state_transition(self.X[:, k], self.U[:, k])
            self.g.append(st - self.X[:, k+1])
            self.lbg.append(np.zeros(self.n_states))
            self.ubg.append(np.zeros(self.n_states))

            # Velocity constraints
            self.g.append(self.U[0, k])
            self.lbg.append(self.v_min)
            self.ubg.append(self.v_max)

            # Angular velocity constraints
            self.g.append(self.U[1, k])
            self.lbg.append(self.omega_min)
            self.ubg.append(self.omega_max)

            # Obstacle avoidance constraints
            for k in range(self.N):
                # Existing cost and constraint definitions

                for obs in self.obstacles:  # Dynamic obstacles
                    distance_to_obstacle = norm_2(self.X[:2, k] - obs.position)
                    self.g.append(distance_to_obstacle)
                    self.lbg.append(self.safety_distance)
                    self.ubg.append(ca.inf)

            # Define the optimization problem
            opts = {'ipopt.print_level': 0, 'print_time': 0}
            self.nlp = {'f': self.cost, 'x': vertcat(self.X, self.U), 'g': vertcat(*self.g)}
            self.solver = nlpsol('solver', 'ipopt', self.nlp, opts)


            
    def solve_mpc(self):
        # Initial condition and reference states
        x0 = vertcat(self.initial_state.reshape((-1, 1)), np.zeros((self.n_controls, self.N)))
        lbg = np.concatenate(self.lbg)
        ubg = np.concatenate(self.ubg)

        # Solve the MPC problem
        sol = self.solver(x0=x0, lbg=lbg, ubg=ubg)
        u_opt = sol['x'][self.n_states * (self.N+1):].reshape((self.n_controls, self.N))

        # Extract the first control input
        linear_velocity = float(u_opt[0, 0])
        angular_velocity = float(u_opt[1, 0])

        return np.array([linear_velocity, angular_velocity])  # linear velocity, angular velocity


# Example initialization
robot = DifferentialDriveRobot(
    robot_id=1,
    initial_state=[0, 0, 0],
    goal_state=[5, 5, np.pi/2],
    N=10,
    max_linear_velocity=0.2,
    max_angular_velocity=0.2,
    safety_distance=0.5,
    obstacles=[]
)






