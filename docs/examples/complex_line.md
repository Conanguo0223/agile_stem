# Complex line

<video src="https://tobias-windisch.de/data/vids/lineflow_complex_line.mov"
       autoplay
       playsinline
       loop
       muted
       style="max-width:100%">
  Sorry, your browser can’t play this video.
</video>

The Complex Line (CL) scenario combines the challenges of multiple previously
explored problems into a unified setting. In this setting, components
and workers must be allocated across a series of k sequential assembly stations.
As in the Waiting Time (WT) scenario, an assembly condition must be maintained:
if a component waits too long before being processed, it becomes invalid. In
addition, the control problem includes aspects from Part Distribution (PD).

## What is optimized?

In CL, the agent must learn a coordinated policy to:

- Distribute components to multiple sequential assembly stations.
- Allocate workers across these stations to balance workloads and maintain throughput.
- Ensure the assembly condition (T<sub>AC</sub>) is satisfied to avoid scrap by
  control the waiting time of the source.

The complexity arises because these tasks are interdependent: an aggressive part distribution policy can overload buffers, causing components to expire before use; on the other hand, slow part distribution delays assembly and reduces output. Similarly, suboptimal worker allocation causes bottlenecks or idle stations.

A well-performing policy must dynamically balance these decisions to ensure a high throughput while minimizing scrap due to expired components.

## Optimization using Lineflow

We used Lineflow to develop and evaluate adaptive control policies for the CL
scenario. RecurrentPPO performed best in this setting, achieving an average reward 
of over 540. As shown in the figure below, we increased the
scrap factor over the course of the training.

![CL Scores](../imgs/rl_benchmark_complex_line_reward.pdf)

In comparison, our self implemented heuristic reached a score of 254. It uses a fixed
worker allocation. The parts are distributed with a prioritization of the most
empty and later buffers. The workers are distributed equally.
