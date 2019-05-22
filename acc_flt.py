import numpy as np

DEFAULT_ACC_FLT_VEL_CONVAR_THRESHOLD = 10.0; 

class AccFilter:
    def __init__(self, max_acc, max_vel, win_ms):
        self.CFG_MAX_ACC = max_acc;
        self.CFG_MAX_VEL = max_vel;
        self.CFG_WINDOW_TIME_MS = win_ms;
        self.CFG_VEL_CONVAR_THRESHOLD = DEFAULT_ACC_FLT_VEL_CONVAR_THRESHOLD; 
        # // The timestamp the stored velocity was generated
        self.state_valid_time_ms = 0
        # // Previous valid velocity
        self.state_valid_vel = np.zeros(3)
        
    def filter(self, vel_3x, variance, curr_time_ms):
        vel = np.array(vel_3x)
        vel_norm = np.linalg.norm(vel)
        if vel_norm > self.CFG_MAX_VEL:
            return False

        # Check acceration
        dt = curr_time_ms - self.state_valid_time_ms;

        print("AccFilter: curr_time_ms {}".format(curr_time_ms))
        print("AccFilter: state_valid_time_ms {}".format(self.state_valid_time_ms))
        print("AccFilter: dt {}".format(dt))
        
        valid = True 

        #Reset filter state if do not have a valid solution for too long
        if dt > self.CFG_WINDOW_TIME_MS:
            print(">>>>>>>>>>>>>>>>>>>>RESET ACC FLT<<<<<<<<<<<<<<<<<<<<")
            self.ResetState()
        else:
            prev_vel_norm = np.linalg.norm(self.state_valid_vel)
            # Check the max acc based on the norm of the velocity vector 
            norm_diff = abs(vel_norm - prev_vel_norm)
            norm_acc = norm_diff * 1e3 / dt
            if norm_acc > self.CFG_MAX_ACC:
                valid = False

            # Check the max acc for each axis 
            axis_acc = np.zeros(3)
            for i in range(3):
                vel_diff = vel[i] - self.state_valid_vel[i]
                axis_acc[i] = vel_diff * 1e3 / dt;
                if abs(axis_acc[i])  > self.CFG_MAX_ACC:
                    valid = False
            print('AccFilter: acc Norm {} N {} E {} D {}'.format(round(norm_acc, 3), round(axis_acc[0], 3), 
                                                      round(axis_acc[1], 3), round(axis_acc[2], 3)))
        
        if not valid:
            print(">>>>>>>>>>>>>>>>>>>>MAX ACC DETECTED<<<<<<<<<<<<<<<<<<<<")
        return valid


    def ResetState(self):
        self.state_valid_vel = np.zeros(3)
        self.state_valid_time_ms = -1

    def UpdateState(self, vel_3x, variance, curr_time_ms):
        variance_norm = np.linalg.norm(variance)
        print("AccFilter: variance_norm {}".format(round(variance_norm, 2)))
        if variance_norm < self.CFG_VEL_CONVAR_THRESHOLD:
            self.state_valid_vel = np.array(vel_3x)
            self.state_valid_time_ms = curr_time_ms
    