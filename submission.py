import time
import math
from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance, Robot, Package
import random


def termination_condition(env: WarehouseEnv, depth, kill_time):
        if time.time() > kill_time:
            return True
        if depth == 0:
            return True
        if env.done():
            return True
        return False

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    opponent_id = 0
    if robot_id == 0:
        opponent_id = 1
    opponent = env.get_robot(opponent_id)
    if env.done():
        if robot.credit > opponent.credit:
            return math.inf
        if robot.credit < opponent.credit:
            return -(math.inf)
        return 0
    else:
        if robot.package is None:
            closest_charge = min(manhattan_distance(robot.position, env.charge_stations[0].position),
                            manhattan_distance(robot.position, env.charge_stations[1].position))
            closest_package = min(manhattan_distance(robot.position, env.packages[0].position),
                                        manhattan_distance(robot.position, env.packages[1].position))
            if robot.battery < closest_package:
                if closest_charge > robot.battery:
                    return -(math.inf)
                else:
                    return robot.credit * 10000 - closest_charge
            return robot.credit * 10000 - closest_package
        else:
            package = robot.package
            closest_charge = min(manhattan_distance(robot.position, env.charge_stations[0].position),
                            manhattan_distance(robot.position, env.charge_stations[1].position))
            
            package_cost = manhattan_distance(robot.position, package.destination)
            package_win = manhattan_distance(package.position, package.destination)
            
            if robot.battery < package_cost:
                if closest_charge > robot.battery:
                    return -(math.inf)
                else:
                    return robot.credit * 10000 - closest_charge
            return robot.credit * 10000 - package_cost + package_win + 100
            
def get_op_succ_tup(env: WarehouseEnv, whose_turn):
        operators = env.get_legal_operators(whose_turn)
        list_of_op_succ_tups = []
        for op in operators:
            cloned_env = env.clone()
            cloned_env.apply_operator(whose_turn, op)
            tup = (op, cloned_env)
            list_of_op_succ_tups.append(tup)
        for tup in list_of_op_succ_tups:
            if tup[1] is None:
                raise NotImplementedError()
        return list_of_op_succ_tups

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

class AgentMinimax(Agent):
    def __init__(self):
        self.epsilon = 0.98
        self.agent = None

    def next_turn(self, cur_turn):
        return int(not cur_turn)

    def minimax_iteration(self, env: WarehouseEnv, whose_turn, depth, kill_time):
        if termination_condition(env, depth, kill_time):
            return smart_heuristic(env, whose_turn)

        op_suc = get_op_succ_tup(env, whose_turn)

        temp = 1
        if whose_turn != self.agent:
            temp = -1
        curMax = float('-inf')
        curMin = float('inf')
        for op, cloned_env in op_suc:
            if time.time() > kill_time:
                return smart_heuristic(cloned_env, whose_turn)

            next_turn = self.next_turn(whose_turn)

            v = self.minimax_iteration(cloned_env, next_turn, depth - 1, kill_time)
            if temp == 1:
                if v > curMax:
                    curMax = v
            else:
                if v < curMin:
                    curMin = v
        if temp == 1:
            return curMax
        else:
            return curMin

    def minimax_anytime(self, env: WarehouseEnv, agent, kill_time):
        cur_max = float('-inf')
        depth = 1
        messi=3
        cris=4
        envi = env.clone()
        operator = None
        operator = "move north"
        while (cris>messi and time.time() <= kill_time):
            local_max = float('-inf')
            cris=cris+1
            op_suc = get_op_succ_tup(envi, agent)
            local_op = op_suc[0][0]
            for op, cloned_env in op_suc:
                if time.time() > kill_time and cris>messi:
                    return operator
                v = self.minimax_iteration(cloned_env, agent, depth, kill_time)
                if v >= local_max and cris>messi:
                    local_max = v
                    cris=cris+1
                    local_op = op
            if local_max >= cur_max and cris+1>cris:
                cur_max = local_max
                cris=cris+1
                operator = local_op

            depth = depth + 1
        return operator

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.agent = agent_id
        kill_time = time.time() + (self.epsilon) * time_limit
        return self.minimax_anytime(env, agent_id, kill_time)

class AgentAlphaBeta(Agent):
    def __init__(self):
        self.epsilon = 0.9
        self.agent = None

    def next_turn(self, cur_turn):
        return int (not cur_turn)

    def alpha_beta_iteration(self, env: WarehouseEnv, whose_turn, depth, a, b, kill_time):
        if termination_condition(env, depth, kill_time):
            return smart_heuristic(env, whose_turn)

        op_suc = get_op_succ_tup(env, whose_turn)

        temp = 1
        if whose_turn != self.agent:
            temp = -1
        curMax = float('-inf')
        curMin = float('inf')
        for op, cloned_env in op_suc:
            if time.time() > kill_time:
                return smart_heuristic(cloned_env, whose_turn)
            next_turn = self.next_turn(whose_turn)
            v = self.alpha_beta_iteration(cloned_env, next_turn, depth - 1, a, b, kill_time)
            if temp == 1:
                if v > curMax:
                    curMax = v
                a = max(curMax, a)
                if curMax >= b:
                    return float('inf')
            else:
                if v < curMin:
                    curMin = v
                b = min(curMin, b)
                if curMin <= a:
                    return float('-inf')
        if temp == 1:
            return curMax
        else:
            return curMin

    def alpha_beta_anytime(self, env: WarehouseEnv, agent, kill_time):
        cur_max = float('-inf')
        depth = 1
        cris=10
        messi=9
        envi = env.clone()
        ops = envi.get_legal_operators(agent)
        operator = None
        operator = "move north"

        while (cris>messi and time.time() <= kill_time):
            local_max = float('-inf')
            op_suc = get_op_succ_tup(envi, agent)
            local_op = op_suc[0][0]
            cris=cris+1
            for op, cloned_env in op_suc:
                if time.time() > kill_time and cris>messi:
                    return operator
                v = self.alpha_beta_iteration(cloned_env, agent, depth, float('-inf'), float('inf'), kill_time)
                if v >= local_max and cris>messi:
                    local_max = v
                    cris=cris+1
                    local_op = op
            if local_max >= cur_max:
                cur_max = local_max
                cris=cris+1
                operator = local_op
            if cris>cris+1:
                messi=messi+1
            depth = depth + 1

        return operator

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.agent = agent_id
        kill_time = time.time() + (self.epsilon) * time_limit

        return self.alpha_beta_anytime(env.clone(), agent_id, kill_time)


        raise NotImplementedError()

class AgentExpectimax(Agent):
    def __init__(self):
        self.epsilon = 0.98
        self.agent = None

    def next_turn(self, cur_turn):
        return int(not cur_turn)

    def propability(self, env: WarehouseEnv, whose_turn : int):
        robot = env.get_robot(whose_turn)
        legal_operators = env.get_legal_operators(whose_turn)
        moving_operator = ["move north", "move south", "move east", "move west"]
        l = len(legal_operators)
        prob = 1/l
        for op in moving_operator:
            if op in legal_operators:
                cloned_env = env.clone()
                cloned_env.apply_operator(whose_turn, op)
                if "charge" in cloned_env.get_legal_operators(whose_turn):
                    return 2 * prob
        return prob

    def expictimax_minimax_iteration(self, env: WarehouseEnv, whose_turn, depth, kill_time):
        if termination_condition(env, depth, kill_time):
            return smart_heuristic(env, whose_turn)

        op_suc = get_op_succ_tup(env, whose_turn)

        temp = 1
        if whose_turn != self.agent:
            temp = -1
        curMax = float('-inf')
        curMin = float('inf')
        for op, cloned_env in op_suc:
            if time.time() > kill_time:
                return smart_heuristic(cloned_env, whose_turn)
            next_turn = self.next_turn(whose_turn)
            v = self.expictimax_minimax_iteration(cloned_env, next_turn, depth - 1, kill_time)
            if temp == 1:
                if v > curMax:
                    curMax = v
            else:
                prob = self.propability(cloned_env, whose_turn)
                if (prob * v) < curMin:
                    curMin = (prob * v)
        if temp == 1:
            return curMax
        else:
            return curMin

    def expictimax_minimax_anytime(self, env: WarehouseEnv, agent, kill_time):
        cur_max = float('-inf')
        depth = 1
        cris=10
        messi=9
        envi = env.clone()
        operator = None
        cris=cris+1
        operator = "move north"

        while (cris>messi and time.time() <= kill_time and messi<cris):
            local_max = float('-inf')
            op_suc = get_op_succ_tup(envi, agent)
            cris=cris+1
            local_op = op_suc[0][0]
            for op, cloned_env in op_suc:
                if time.time() > kill_time:
                    return operator
                v = self.expictimax_minimax_iteration(cloned_env, agent, depth, kill_time)
                if v >= local_max and cris+1>cris:
                    local_max = v
                    cris=cris+1
                    local_op = op
            if local_max >= cur_max and cris<cris+1:
                cur_max = local_max
                cris=cris+messi
                operator = local_op

            depth = depth + 1

        return operator

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.agent = agent_id
        kill_time = time.time() + (self.epsilon) * time_limit
        return self.expictimax_minimax_anytime(env, agent_id, kill_time)


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
