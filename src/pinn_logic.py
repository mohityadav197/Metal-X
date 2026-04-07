import numpy as np

class MetallurgicalTeacher:
    def __init__(self):
        self.temp_limit = 195.0 
        
    def calculate_penalty(self, data):
        penalty = 0.0
        mg = data.get('mg', 0)
        si = data.get('si', 0)
        temp = data.get('temperature', 0)
        
        # 1. Stoichiometry (Basic Ratio Check)
        ratio = mg / (si + 1e-6)
        if ratio < 1.0 or ratio > 2.5:
            penalty += 0.4
            
        # 2. Thermal Check (Over-aging)
        if temp > self.temp_limit:
            penalty += 0.5
            
        return float(np.clip(penalty, 0.0, 1.0))