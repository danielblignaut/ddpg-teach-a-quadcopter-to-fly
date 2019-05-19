import numpy as np
from physics_sim import PhysicsSim

class TakeOffTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 100
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 1.]) 

        self.initial_distance = self.get_distance_to_target(self.sim.pose[:3],target_pos)

    def get_distance_to_target(self, pos, target) :
        distance_to_target = ((pos[0] - target[0]) ** 2 + (pos[1] - target[1]) ** 2 + (pos[2] - target[2]) ** 2) ** 0.5
        return distance_to_target

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        absolute_distance_constant = -5
        survival_constant = -0.5
        #x_distance_constant = -2
        #y_distance_constant = -2
        #z_distance_constant = -2
        radian = 0.0174533
        distance_to_target = self.get_distance_to_target(self.sim.pose[:3], self.target_pos[:3])
        reward = 0.5 * (self.initial_distance - distance_to_target)
        
        if(self.sim.pose[2] > 0) :
            reward += 200

        #constant rewaard
        #reward = 100

        #penalty for general distance to tareget
        #reward += absolute_distance_constant * abs(distance_to_target)

        #penalty for euler angle changes... we want to maintain stability
        #reward += -2 * abs(self.sim.pose[3:6]).sum()

        #penalty for big euler angle changes... we want to maintain stability
        #reward += -5 * abs(self.sim.angular_v).sum()

        

        #reward = x_distance_constant * abs(self.sim.pose[0] - self.target_pos[0])
        #reward += y_distance_constant * abs(self.sim.pose[1] - self.target_pos[1]) 
        #reward += z_distance_constant * abs(self.sim.pose[2] - self.target_pos[2])
        #reward += survival_constant * self.survived_number_of_rounds
        #reward += -1 * (self.sim.pose[3] - 90 * radian)
        #reward += -1 * (self.sim.pose[4] - 90 * radian)
        #reward += -1 * (self.sim.pose[5] - 90 * radian)

        #print("distance reward: {:7.3f}, survival reward: {:7.3f}, ".format(penalty_constant * distance_to_target, survival_constant * self.survived_number_of_rounds))
        #TODO: add a penalty for time to promote completion speed
        #TODO: add a penalty / reward for slower velocity over time when within a certain area of reward that outweighs time constraint
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)

        if(self.get_distance_to_target(self.sim.pose[:3],self.target_pos) < .1) :
            done = True
        
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    
    def get_target_pos(self) :
        return self.target_pos