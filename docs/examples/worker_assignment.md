# Worker Assignment

<video src="https://tobias-windisch.de/data/vids/lineflow_worker_assignment.mov"
       autoplay
       playsinline
       loop
       muted
       style="max-width:100%">
  Sorry, your browser can’t play this video.
</video>

In many production lines, manual tasks can be completed more efficiently 
when multiple workers collaborate. This example seeks to identify the optimal 
distribution of a limited workforce across various stations, each with different 
processing times. In addition, it is assumed that the processing time at each station depends on
the number of workers assigned. In our experiments, adding one more worker reduces 
the processing time by approximately 74%.

## What is optimized?
The stations are linked by buffers, and the overall production speed is determined 
by the slowest station. Therefore, the goal is to allocate workers in such a 
way that the maximum processing time across all stations is minimized. 
This allocation process is constrained by factors such as delays caused by the 
movement of workers between stations. If workers are assigned in a way that maximizes 
the production time of the slowest station, the expected reward is also maximized.

## Optimization using Lineflow
We can get the best results by using the actor-critic method A2C with an averaged reward of 
278, surpassing all other methods and roughly reaching the reward optimum of 287. 
![Worker Assignment rl](../imgs/rl_benchmark_worker_assignment_N_3_reward.png)

## Verification of the optimization
We compare the results given by LineFlow with the theoretical optimal solution in 
the following paragraph.
This problem can be framed as an optimization challenge, where the objective is 
to minimize the maximum processing time by determining the optimal worker 
distribution across stations. This requires solving a max-min problem, which is 
computationally complex due to its inherent difficulty and the need for integer 
partitioning. Additionally, finding the best distribution in a dynamic environment 
is even more challenging, as it involves accurately estimating the parameters for 
each station. To check the worker distribution calculated by LineFlow, all possible 
worker allocations are listed to evaluate them empirically. As mentioned above,
the line can reach a reward of 287 with the optimal worker allocation.
For a detailed calculation of the worker assignment, see *(Link to Lineflow paper)*.

![Worker Assignment optimum](../imgs/worker_assignment_empirical_n=3.png)

## Code
```python
--8<-- "lineflow/examples/worker_assignment.py"
```
