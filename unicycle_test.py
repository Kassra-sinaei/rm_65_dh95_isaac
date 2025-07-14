import crocoddyl
import numpy as np

# Problem settings
dt       = 0.1
T        = 50
x0       = np.array([0., 0., 0.])
x_target = np.array([5., 5., np.pi/2])
v_min, v_max         = 0.0, 1.0
omega_min, omega_max = -np.pi/4, np.pi/4

#
# 1) Discrete‐time unicycle action model
#
class ActionDataUnicycle(crocoddyl.ActionDataAbstract):
    """Holds both the standard fields and a slot for the cost‐model’s data."""
    def __init__(self, model: crocoddyl.ActionModelAbstract):
        super().__init__(model)
        self.cost_data = None

class ActionModelUnicycle(crocoddyl.ActionModelAbstract):
    """
    Discrete‐time unicycle: x_{k+1} = x_k + dt·[v cosθ, v sinθ, ω]ᵀ,
    with costs defined via a shared CostModelSum.
    """
    def __init__(self, cost_model: crocoddyl.CostModelSum, dt: float):
        state = crocoddyl.StateVector(3)
        nu    = 2
        nr    = cost_model.nr
        super().__init__(state, nu, nr)
        self.cost_model = cost_model
        self.dt         = dt

    def createData(self):
        # Allocate the base data (includes data.shared for costs/contacts)
        data = ActionDataUnicycle(self)
        collector = crocoddyl.DataCollectorAbstract()
        data.cost_data = self.cost_model.createData(collector)
        return data

    def calc(self, data, x, u=None):
        θ = x[2]
        if u is None:
            # Terminal node: no control input, so just propagate state and cost
            v, ω = 0., 0.
        else:
            v, ω = u
        # 1) next state
        data.xnext = x + self.dt * np.array([
            v * np.cos(θ),
            v * np.sin(θ),
            ω
        ])
        # 2) cost
        if u is None:
            self.cost_model.calc(data.cost_data, x, np.zeros(2))
        else:
            self.cost_model.calc(data.cost_data, x, u)
        data.cost = data.cost_data.cost

    def calcDiff(self, data, x, u=None):
        θ = x[2]
        if u is None:
            v = 0.
            u = np.zeros(2)
        else:
            v, _ = u
        c, s = np.cos(θ), np.sin(θ)
        # a) dynamics derivatives
        Fx = np.eye(3)
        Fx[0,2] = -self.dt * v * s
        Fx[1,2] =  self.dt * v * c
        Fu = self.dt * np.array([[c, 0.],
                                [s, 0.],
                                [0., 1.]])
        data.Fx, data.Fu = Fx, Fu

        # b) cost derivatives (populates cost_data.Lx, Lu, Lxx, …)
        self.cost_model.calcDiff(data.cost_data, x, u)
        data.Lx   = data.cost_data.Lx
        data.Lu   = data.cost_data.Lu
        data.Lxx  = data.cost_data.Lxx
        data.Lxu  = data.cost_data.Lxu
        data.Luu  = data.cost_data.Luu

#
# 2) Build shared cost model
#
state = crocoddyl.StateVector(3)
cmodel = crocoddyl.CostModelSum(state, 2)
#   – state tracking
res_s = crocoddyl.ResidualModelState(state, x_target, 2)
cmodel.addCost("stateReg",
               crocoddyl.CostModelResidual(state, res_s),
               weight=1.0)
#   – control regularization
res_u = crocoddyl.ResidualModelControl(state, 2)
cmodel.addCost("ctrlReg",
               crocoddyl.CostModelResidual(state, res_u),
               weight=1e-2)

#
# 3) Instantiate one ActionModelUnicycle per node, enforce box limits
#
running_models = []
for k in range(T):
    # use the same cmodel for running; at terminal you might up‐weight stateReg
    am = ActionModelUnicycle(cmodel, dt)
    am.u_lb = np.array([v_min, omega_min])
    am.u_ub = np.array([v_max, omega_max])
    running_models.append(am)

#
# 4) Solve with BoxFDDP
#
problem = crocoddyl.ShootingProblem(x0, running_models[:-1], running_models[-1])
# solver  = crocoddyl.SolverBoxFDDP(problem)
solver  = crocoddyl.SolverDDP(problem) # This won't explicitly enforce the box limits

# warm‐start
us_init = [np.zeros(2) for _ in range(T-1)]
xs_init = solver.problem.rollout(us_init)

solver.solve(xs_init, us_init, maxiter=100)
print("Solved! Final state:", solver.xs[-1])