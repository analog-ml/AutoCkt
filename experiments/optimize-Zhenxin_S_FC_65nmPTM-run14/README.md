![alt text](image.png)

-  **Directory:** **~/code/analog-ml/AutoCkt-optimize-Zhenxin_S_FC_65nmPTM-run14** 
	- **hostname**: server-2
	- forked from `AutoCkt-optimize-Zhenxin_S_FC_65nmPTM-run12`
	-  <span style="color:rgb(0, 176, 80)">branch</span>: `AutoCkt-optimize-Zhenxin_S_FC_65nmPTM-run14`
	- **purpose**: reduce the number tunnable parameters: `W` and `CC` only
	- <span style="color:rgb(112, 48, 160)">new things added:</span>
		- reduce the tunable parameters to 11.
	- <span style="color:rgb(255, 0, 0)">things have modified:</span>
		- change unit to "nm"
	- <span style="color:rgb(255, 192, 0)">tensorboard</span>:  `PPO_Zhenxin_S_FC_0_2025-09-04_16-10-519x116nc6`
	- <span style="color:rgb(255, 0, 0)">errors</span>:
	- <span style="color:rgb(0, 112, 192)">solutions</span>:
	- **resume**
	- **completion notes**
		- matches expectation: yes
		- After 800k (7 hours 30 min on server-2) reached `ray/tune/episode_reward_mean=0.0`