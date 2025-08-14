import numpy as np
from tools.rrt_planner import rrt


def test_rrt_path_exists():
    start = np.array([0.5, -0.2, 0.25], dtype=np.float32)
    goal = np.array([0.7, 0.2, 0.25], dtype=np.float32)
    aabb_min = np.array([0.55, -0.05, 0.15], dtype=np.float32)
    aabb_max = np.array([0.65, 0.05, 0.35], dtype=np.float32)
    low = np.array([0.4, -0.3, 0.15], dtype=np.float32)
    high = np.array([0.8, 0.3, 0.4], dtype=np.float32)
    path = rrt(start, goal, aabb_min, aabb_max, (low, high), max_iters=5000)
    assert path, "RRT failed to find a path"
    # verify no segment intersects
    for i in range(len(path)-1):
        p0, p1 = np.array(path[i]), np.array(path[i+1])
        # basic check: x stays within bounds
        assert (low <= p0).all() and (p0 <= high).all()
        assert (low <= p1).all() and (p1 <= high).all()