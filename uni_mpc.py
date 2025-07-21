import numpy as np
import casadi as ca

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import pinocchio as pin

class MPCController:
    def __init__(self, horizon=10, dt=0.01, Q=None, R=None, Qf=None, 
                 v_max=1.0, omega_max=1.0, x_bounds=None, y_bounds=None):
        """
        MPC Controller for differential drive mobile robot using CasADi
        
        Args:
            horizon: prediction horizon
            dt: sampling time
            Q: state cost matrix (3x3)
            R: input cost matrix (2x2)  
            Qf: final state cost matrix (3x3)
            v_max: maximum linear velocity
            omega_max: maximum angular velocity
            x_bounds: workspace x bounds [x_min, x_max]
            y_bounds: workspace y bounds [y_min, y_max]
        """
        self.N = horizon
        self.dt = dt
        
        # Cost matrices
        self.Q = Q if Q is not None else np.diag([10.0, 10.0, 10.0])  # [x, y, theta]
        self.R = R if R is not None else np.diag([1.0, 2.0])         # [v, omega]
        self.Qf = Qf if Qf is not None else np.diag([100.0, 100.0, 100.0])
        
        # Control constraints
        self.v_max = v_max
        self.omega_max = omega_max
        
        # State constraints
        self.x_bounds = x_bounds if x_bounds else [-100.0, 100.0]
        self.y_bounds = y_bounds if y_bounds else [-100.0, 100.0]
        
        # Setup optimization problem
        self.setup_optimization_problem()
        
    def unicycle_dynamics(self, state, control):
        """
        CasADi-compatible unicycle dynamics
        
        Args:
            state: [x, y, theta] (CasADi MX)
            control: [v, omega] (CasADi MX)
            
        Returns:
            state_dot: [x_dot, y_dot, theta_dot] (CasADi MX)
        """
        
        x_dot = control[0] * ca.cos(state[2])
        y_dot = control[0] * ca.sin(state[2])
        theta_dot = control[1]
        
        return ca.vertcat(x_dot, y_dot, theta_dot)
    
    def integrate_dynamics(self, state, control):
        """
        Integrate dynamics using Euler method (CasADi compatible)
        
        Args:
            state: current state [x, y, theta]
            control: control input [v, omega]
            
        Returns:
            next_state: state at next time step
        """
        state_dot = self.unicycle_dynamics(state, control)
        next_state = state + self.dt * state_dot
        return next_state
    
    def setup_optimization_problem(self):
        """
        Setup the CasADi optimization problem structure
        """
        # Create optimizer
        self.opti = ca.Opti()
        
        # Decision variables
        self.U = self.opti.variable(2, self.N)  # Controls [v, omega] for each time step
        self.X = self.opti.variable(3, self.N)  # States [x, y, theta] for each time step
        
        # Parameters (will be set at each control call)
        self.x0 = self.opti.parameter(3)  # Initial state
        self.x_ref = self.opti.parameter(3, self.N + 1)  # Reference trajectory
        
        # Convert numpy matrices to CasADi
        Q_ca = ca.DM(self.Q)
        R_ca = ca.DM(self.R)
        Qf_ca = ca.DM(self.Qf)
        
        # Cost function
        cost = 0
        
        # Stage costs (state tracking + control effort)
        for k in range(self.N):
            # State tracking cost
            if k < self.N - 1:
                state_error = self.X[:, k] - self.x_ref[:, k]
                cost += state_error.T @ Q_ca @state_error
            
            # Control effort cost
            cost += self.U[:, k].T @ R_ca @ self.U[:, k]
        
        # Terminal cost
        final_state_error = self.X[:, -1] - self.x_ref[:, -1]
        cost += final_state_error.T @ Qf_ca @ final_state_error
        
        self.opti.minimize(cost)
        
        # Dynamics constraints
        for k in range(self.N):
            if k == 0:
                # Initial condition
                x_next = self.integrate_dynamics(self.x0, self.U[:, k])
            else:
                # Dynamics propagation
                x_next = self.integrate_dynamics(self.X[:, k-1], self.U[:, k])
            
            self.opti.subject_to(self.X[:, k] == x_next)
        
        # Control constraints
        self.opti.subject_to(self.opti.bounded(-self.v_max, self.U[0, :], self.v_max))
        self.opti.subject_to(self.opti.bounded(-self.omega_max, self.U[1, :], self.omega_max))
        
        # State constraints
        self.opti.subject_to(self.opti.bounded(self.x_bounds[0], self.X[0, :], self.x_bounds[1]))
        self.opti.subject_to(self.opti.bounded(self.y_bounds[0], self.X[1, :], self.y_bounds[1]))
        self.opti.subject_to(self.opti.bounded(-ca.pi, self.X[2, :], ca.pi))
        
        # Solver options
        # opts = {
        #     'ipopt.print_level': 0,
        #     'ipopt.max_iter': 200,
        #     'ipopt.tol': 1e-4,
        #     'ipopt.acceptable_tol': 1e-3,
        #     'print_time': 0,
        #     'ipopt.warm_start_init_point': 'yes'
        # }
        
        # self.opti.solver('ipopt', opts)
        opts = {
            'qpsol': 'qrqp',
            'print_header': False,
            'print_iteration': False,
            'print_time': False,
            'qpsol_options': {
                'print_iter': False,
                'print_header': False
            }
        }
        self.opti.solver('sqpmethod', opts)
        
        # Initialize solver (for warm starting)
        self.previous_solution = None
    
    def control(self, current_state, reference_trajectory):
        """
        Solve MPC optimization and return control input
        
        Args:
            current_state: current robot state [x, y, theta]
            reference_trajectory: desired trajectory (N+1 x 3 array)
            
        Returns:
            optimal_control: optimal control input [v, omega]
            success: boolean indicating if optimization succeeded
        """
        try:
            # Set parameters
            self.opti.set_value(self.x0, current_state)
            self.opti.set_value(self.x_ref, reference_trajectory.T)
            
            # Warm start with previous solution if available
            if self.previous_solution is not None:
                # Shift previous solution for warm start
                U_warmstart = np.zeros((2, self.N))
                X_warmstart = np.zeros((3, self.N))
                
                if self.N > 1:
                    # Shift control inputs
                    U_warmstart[:, :-1] = self.previous_solution['U'][:, 1:]
                    U_warmstart[:, -1] = self.previous_solution['U'][:, -1]  # Repeat last control
                    
                    # Shift states
                    X_warmstart[:, :-1] = self.previous_solution['X'][:, 1:]
                    # Propagate last state
                    last_state = self.previous_solution['X'][:, -1]
                    last_control = U_warmstart[:, -1]
                    X_warmstart[:, -1] = self._integrate_dynamics_numpy(last_state, last_control)
                else:
                    U_warmstart = self.previous_solution['U']
                    X_warmstart = self.previous_solution['X']
                
                self.opti.set_initial(self.U, U_warmstart)
                self.opti.set_initial(self.X, X_warmstart)
            else:
                # Cold start with reasonable initial guess
                self._set_initial_guess(current_state, reference_trajectory)
            
            # Solve optimization
            sol = self.opti.solve()
            
            # Extract solution
            U_opt = sol.value(self.U)
            X_opt = sol.value(self.X)
            
            # Store solution for next warm start
            self.previous_solution = {'U': U_opt, 'X': X_opt}
            
            # Return first control input (receding horizon principle)
            optimal_control = U_opt[:, 0]
            
            return optimal_control, True
            
        except Exception as e:
            print(f"MPC optimization failed: {e}")
            # Return safe control input
            return np.array([0.0, 0.0]), False
    
    def _integrate_dynamics_numpy(self, state, control):
        """
        Numpy version of dynamics integration for warm starting
        """
        x, y, theta = state
        v, omega = control
        
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = omega
        
        state_dot = np.array([x_dot, y_dot, theta_dot])
        next_state = state + self.dt * state_dot
        return next_state
    
    def _set_initial_guess(self, current_state, reference_trajectory):
        """
        Set initial guess for optimization variables
        """
        # Simple initial guess for controls
        U_guess = np.zeros((2, self.N))
        
        # Initial guess for states - simple forward integration
        X_guess = np.zeros((3, self.N))
        state = current_state.copy()
        
        for k in range(self.N):
            # Use small forward velocity as initial guess
            control_guess = np.array([0.1, 0.0])
            state = self._integrate_dynamics_numpy(state, control_guess)
            X_guess[:, k] = state
            U_guess[:, k] = control_guess
        
        self.opti.set_initial(self.U, U_guess)
        self.opti.set_initial(self.X, X_guess)
    
    def generate_reference_trajectory(self, current_state, target_state):
        """
        Generate a simple reference trajectory from current to target state
        
        Args:
            current_state: starting state [x, y, theta]
            target_state: goal state [x, y, theta]
            
        Returns:
            reference: trajectory array (N+1 x 3)
        """
        reference = np.zeros((self.N + 1, 3))
        reference[0] = current_state
        
        # Linear interpolation for position, handle angle separately
        for k in range(1, self.N + 1):
            alpha = k / self.N
            
            # Linear interpolation for x, y
            reference[k, 0] = (1 - alpha) * current_state[0] + alpha * target_state[0]
            reference[k, 1] = (1 - alpha) * current_state[1] + alpha * target_state[1]
            
            # Handle angle interpolation (shortest path)
            angle_diff = target_state[2] - current_state[2]
            # Wrap angle difference to [-pi, pi]
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            reference[k, 2] = current_state[2] + alpha * angle_diff
            
            # Wrap final angle to [-pi, pi]
            reference[k, 2] = np.arctan2(np.sin(reference[k, 2]), np.cos(reference[k, 2]))
        
        return reference
    
    def get_prediction(self):
        """
        Get the predicted trajectory from the last optimization
        
        Returns:
            predicted_states: predicted state trajectory (N x 3)
            predicted_controls: predicted control trajectory (N x 2)
        """
        if self.previous_solution is not None:
            return self.previous_solution['X'].T, self.previous_solution['U'].T
        else:
            return None, None
    
    def plot_ref(self, goal_state, current_state, reference_traj):
        """
        Simple plot of x, y, theta reference trajectory
        """
        # Extract trajectory data
        x_ref = reference_traj[:, 0]
        y_ref = reference_traj[:, 1]
        theta_ref = reference_traj[:, 2]
        time_steps = np.arange(len(reference_traj))
        
        # Create figure with 3 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot X trajectory
        ax1.plot(x_ref, y_ref, 'b-', linewidth=2, marker='o', markersize=3)
        ax1.set_xlabel('X Position [m]')
        ax1.set_ylabel('Y Position [m]')
        ax1.set_title('Reference X Trajectory')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot Y trajectory
        ax2.plot(time_steps, reference_traj[:, 2], 'g-', linewidth=2, marker='o', markersize=3)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Heading Angle [rad]')
        ax2.set_title('Reference Yaw')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save and show
        timestamp = int(time.time())
        plt.savefig(f'/tmp/simple_ref_traj_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(5)  # Show for 1 second
        plt.close()
        pass

    def reset_solver(self):
        """
        Reset the solver (clear warm start)
        """
        self.previous_solution = None

def plot_trajectory(states, reference, controls, title="MPC Trajectory"):
    """
    Plot the robot trajectory, reference, and control inputs
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract coordinates
    x_traj = [s[0] for s in states]
    y_traj = [s[1] for s in states]
    theta_traj = [s[2] for s in states]
    
    x_ref = reference[:, 0]
    y_ref = reference[:, 1]
    
    # Plot 1: XY trajectory
    ax1.plot(x_ref, y_ref, 'r--', linewidth=2, label='Reference', alpha=0.7)
    ax1.plot(x_traj, y_traj, 'b-', linewidth=2, label='Actual')
    ax1.plot(x_traj[0], y_traj[0], 'go', markersize=8, label='Start')
    ax1.plot(x_traj[-1], y_traj[-1], 'ro', markersize=8, label='End')
    
    # Add robot orientation arrows
    for i in range(0, len(states), max(1, len(states)//10)):
        dx = 0.3 * np.cos(theta_traj[i])
        dy = 0.3 * np.sin(theta_traj[i])
        ax1.arrow(x_traj[i], y_traj[i], dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)
    
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('XY Trajectory')
    ax1.axis('equal')
    
    # Plot 2: States vs time
    time_steps = np.arange(len(states)) * 0.05  # Assuming dt=0.05
    ax2.plot(time_steps, x_traj, 'b-', label='X position')
    ax2.plot(time_steps, y_traj, 'g-', label='Y position')
    ax2.plot(time_steps, theta_traj, 'r-', label='Theta [rad]')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('State values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('States vs Time')
    
    # Plot 3: Control inputs
    if len(controls) > 0:
        control_time = np.arange(len(controls)) * 0.05
        v_controls = [c[0] for c in controls]
        omega_controls = [c[1] for c in controls]
        
        ax3.plot(control_time, v_controls, 'b-', linewidth=2, label='Linear velocity [m/s]')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Linear velocity [m/s]')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Linear Velocity')
        
        ax4.plot(control_time, omega_controls, 'r-', linewidth=2, label='Angular velocity [rad/s]')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Angular velocity [rad/s]')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Angular Velocity')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=True)
    
def simulate_mpc_trajectory(mpc, initial_state, target_state, max_time=10.0, tolerance=0.1):
    """
    Simulate the complete MPC trajectory
    """
    current_state = initial_state.copy()
    states = [current_state.copy()]
    controls = []
    
    dt = mpc.dt
    time_elapsed = 0.0
    
    print(f"Starting simulation from {initial_state} to {target_state}")
    
    while time_elapsed < max_time:
        # Check if we've reached the target
        distance_error = np.linalg.norm(current_state[:2] - target_state[:2])
        angle_error = abs(np.arctan2(np.sin(current_state[2] - target_state[2]), 
                                    np.cos(current_state[2] - target_state[2])))
        
        if distance_error < tolerance and angle_error < 0.1:
            print(f"Target reached in {time_elapsed:.2f} seconds!")
            break
        
        # Generate reference trajectory
        reference_traj = mpc.generate_reference_trajectory(current_state, target_state)
        
        # Get control input
        control_input, success = mpc.control(current_state, reference_traj)
        
        if not success:
            print(f"MPC failed at time {time_elapsed:.2f}")
            break
        
        # Apply control and integrate dynamics
        current_state = mpc._integrate_dynamics_numpy(current_state, control_input)
        
        # Store data
        states.append(current_state.copy())
        controls.append(control_input.copy())
        
        time_elapsed += dt
        
        # Print progress every second
        if len(states) % int(1.0/dt) == 0:
            print(f"Time: {time_elapsed:.1f}s, State: [{current_state[0]:.2f}, {current_state[1]:.2f}, {current_state[2]:.2f}], "
                    f"Control: [{control_input[0]:.2f}, {control_input[1]:.2f}], Distance: {distance_error:.3f}")
    
    return states, controls, reference_traj

def test_mpc():
    # Create MPC controller
    mpc = MPCController(
        horizon=50,
        dt=0.01,
        Q=np.diag([25.0, 25.0, 0.1]),
        R=np.diag([1.0, 0.01]),
        Qf=np.diag([50.0, 50.0, 0.2]),
        v_max=1.0,
        omega_max=1.0
    )
    
    # Initial and target states
    current_state = np.array([0.0, 0.0, 0.0])
    target_state = np.array([5.0, 3.0, np.pi/2])
    
    # Test single control computation
    # reference_traj = mpc.generate_reference_trajectory(current_state, target_state)
    reference_traj = target_state * np.ones((mpc.N + 1, 3))  # Simple constant reference
    control_input, success = mpc.control(current_state, reference_traj)
    
    if success:
        print(f"Optimal control: v = {control_input[0]:.3f}, omega = {control_input[1]:.3f}")
        
        # Run full simulation
        states, controls, reference = simulate_mpc_trajectory(
            mpc, current_state, target_state, max_time=10.0
        )
        
        if len(states) > 1:
            plot_trajectory(states, reference, controls, 
                          title="MPC Trajectory - Single Test")
    else:
        print("Optimization failed")
    pass


class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node')
        self.publisher = self.create_publisher(Twist, '/isaac_sim/cmd_vel', 1)
        self.subscription = self.create_subscription(Odometry, '/isaac_sim/odom', self.odom_callback, 1)
        self.mpc = MPCController(
            horizon=50,
            dt=0.01,
            Q=np.diag([25.0, 25.0, 0.1]),
            R=np.diag([1.0, 0.01]),
            Qf=np.diag([50.0, 50.0, 0.2]),
            v_max=1.0,
            omega_max=1.0
        )
        self.current_state = np.array([0.0, 0.0, 0.0])
        self.target_state = np.array([1.0, 0.5, 0])
        self.timer = self.create_timer(0.01, self.control_loop)

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        rotation_matrix = pin.Quaternion( orientation.w, orientation.x, orientation.y, orientation.z).toRotationMatrix() 
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        self.current_state = np.array([position.x, position.y, yaw])

    def control_loop(self):
        # reference_traj = self.target_state * np.ones((self.mpc.N + 1, 3))  # Simple constant reference
        reference_traj = self.mpc.generate_reference_trajectory(self.current_state, self.target_state)
        control_input, success = self.mpc.control(self.current_state, reference_traj)
        if success:
            twist_msg = Twist()
            twist_msg.linear.x = control_input[0]
            twist_msg.angular.z = control_input[1]
            self.publisher.publish(twist_msg)
            print(f"Current State: {self.current_state.T}, Control: {control_input}")
            print(f"Error State: {self.current_state.T - self.target_state.T}")
        else:
            self.get_logger().warn("MPC optimization failed, stopping robot.")
            self.publisher.publish(Twist())

def jetbot_test(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    # test_mpc()  
    jetbot_test()