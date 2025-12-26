import numpy as np
from tools.geom_tools import build_curve_bridge

# Test bridge
start = np.array([10.0, 20.0])
end = np.array([30.0, 40.0])
start_tan = np.array([1.0, 0.5])
end_tan = np.array([0.8, 0.6])

bridge = build_curve_bridge(start, end, start_tan, end_tan)
print(f"Bridge shape: {bridge.shape}")
print(f"Bridge type: {type(bridge)}")
print(f"Bridge preview:\n{bridge[:3]}")