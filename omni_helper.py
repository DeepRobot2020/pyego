import numpy as np
from numpy import linalg as LA

``
class OmniCameraModel:
    def __init__(self, fu = 0, fv = 0, cu = 320, cv = 240, xi = 0, cam_model = 'pinhole', width = 640, height = 480):
        self.cam_model = cam_model
        self._fu = fu
        self._fv = fv
            
        self._cu = cu
        self._cv = cv
        self._xi = xi
                
        self._fov_parameter = 0
        self._distortion = None
        
        self._recip_fu = 1.0 / self._fu;
        self._recip_fv = 1.0 / self._fv;
        self._fu_over_fv = self._fu / self._fv;
        self._one_over_xixi_m_1 = 1.0 / (self._xi * self._xi - 1)
        self._fov_parameter = self._xi if self._xi <= 1.0 else 1 / _xi;
        
        
        self.width = width
        self.height = height
        self.K = None
        
    def ru(self):
        return self.width
    
    def rv(self):
        return self.height
    
    def xi(self):
        return self._xi
    
    def isValid(keypoint):
        return keypoint[0] >= 0 && keypoint[0] < ru() && keypoint[1] >= 0 && keypoint[1] < rv()
    
    def isUndistortedKeypointValid(self, rho2_d):
        return self.xi() <= 1.0 or rho2_d <= self._one_over_xixi_m_1;
    
    def euclideanToKeypoint(self, p):
        d = LA.norm(p)
        
        # Check if point will lead to a valid projection
        if p[2] <= -self._fov_parameter * d
            return None

        outKeypoint = np.zeros([2, 1])
        
        rz = 1.0 / (p[2] + self._xi * d)
        
        outKeypoint[0] = p[0] * rz
        outKeypoint[1] = p[1] * rz

        self._distortion.distort(outKeypoint)
        
        outKeypoint[0] = self._fu * outKeypoint[0] + self._cu
        outKeypoint[1] = self._fv * outKeypoint[1] + self._cv
        
        if self.isValid(outKeypoint):
            return outKeypoint
        
        return None
  
    def keypointToEuclidean(self, keypoint):
        outPoint = np.zeros([3, 1])
        
        # Unproject...
        outPoint[0] = self._recip_fu * (keypoint[0] - self._cu);
        outPoint[1] = self._recip_fv * (keypoint[1] - self._cv);

        # Re-distort
        self._distortion.undistort(outPoint);

        rho2_d = outPoint[0] * outPoint[0] + outPoint[1] * outPoint[1];

        if self.isUndistortedKeypointValid(rho2_d) is False:
            return None;

        _xi = self._xi
        outPoint[2] = 1 - _xi * (rho2_d + 1) / (_xi + sqrt(1 + (1 - _xi * _xi) * rho2_d))
        
        return outPoint
    
    

    


    