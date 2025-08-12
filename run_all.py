import os

from tasks.task1_reach import run as run1
from tasks.task2_path_tracing import run as run2
from tasks.task3_pick_place import run as run3
from tasks.task4_obstacle_pick_place import run as run4
from tasks.task5_stacking import run as run5


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    run1("outputs/task1_reach.mp4")
    run2("outputs/task2_path_tracing.mp4")
    run3("outputs/task3_pick_place.mp4")
    run4("outputs/task4_obstacle_pick_place.mp4")
    run5("outputs/task5_stacking.mp4")