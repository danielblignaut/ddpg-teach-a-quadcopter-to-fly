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
        self.action_repeat = 10

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
        relative_distance_constant = 4
        angular_velociy_constant = -1

        distance_to_target = self.get_distance_to_target(self.sim.pose[:3], self.target_pos[:3])
        reward = relative_distance_constant * np.tanh(self.initial_distance - distance_to_target)

        reward += angular_velociy_constant * abs(self.sim.angular_v[0]) + abs(self.sim.angular_v[1]) + abs(self.sim.angular_v[1])


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