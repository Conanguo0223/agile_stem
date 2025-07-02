# Part Distribution


<video src="https://tobias-windisch.de/data/vids/lineflow_partdistribution.mov"
       autoplay
       playsinline
       loop
       muted
       style="max-width:100%">
  Sorry, your browser can’t play this video.
</video>


This example focuses on a specific scheduling problem in manufacturing. 
Components need to be distributed across multiple parallel 
processes by a single switch (SwitchD) in an optimal manner. Additionally, another 
switch (SwitchF) is responsible for collecting the outputs from these processes and 
sending them to a final destination. These parallel processes differ in 
terms of the processing times. As a result, they need a different number of 
parts distributed by the switch in order to work optimally.

Within a given timeframe, the maximum number of parts that a process can 
produce is determined by ensuring a continuous supply of components without 
any interruptions. If the processing times of the source and the final destination 
are negligible compared to those of the processes, the total number of parts 
produced by the system is simply the sum of the parts produced by each individual 
process. This provides an estimate of the maximum production capacity of the system.

## What is optimized?
We optimize the distribution of the parts by the switch (SwitchD) to the stations, considering 
the processing times of all processing stations. In addition, collecting the parts that 
have been processed and can now be pushed to the final station at the end of the line 
needs to be optimized (SwitchF). A policy implemented at both switches, that assigns 
components by placing them in the least filled buffer, while prioritizing faster
processes, and retrieving them from the most 
filled buffer seems like the best approach for this problem. We examine the case with k=5 
stations. Processing times vary with &sum;<sub>i</sub><sup>k</sup> 10+10*i + exp(0.2).
The reward can be maximized by optimizing the part distribution of both switches.

## Optimization using Lineflow
We validated the effectiveness of Lineflow algorithms by comparing them to the rewards 
achieved by the greedy approach. Our optimal agent achieved rewards up to 738, as shown 
in the plot below. Only the policy based methods PPO and TRPO ensure a proper switch distribution.
![Part distrib empirical n5 reward](../imgs/rl_benchmark_part_distribution_N_5_reward.png)

## Verification of the optimization
As mentioned above, a greedy scheduling policy is implemented at both switches. This policy 
prioritizes sending components to the least filled buffer and retrieving 
components from the most filled buffer, ensuring an optimal distribution and therefore the 
maximum number of parts produced. 
A comparison of the component distributions between the greedy policy and the optimal 
distribution for 5 parallel processing cells is given in the following figure. 
For a detailed calculation and proof of the optimal part distribution, see *(Link to Lineflow paper)*.
![Part distrib empirical n5](../imgs/part_distributions_empirical_n=5.png)

## Code

```python
--8<-- "lineflow/examples/multi_process.py"
```
