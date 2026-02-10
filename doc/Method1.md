
# Method 1 description 

The first methodology on a combination of several minimization subproblems, and a scoring algorithm. 

The process steps are as follow : 

1. First leg table :  The first step consists in building a table of 400 lines describing solution of the first leg.

The varied parameters are :
		- The arrival planet position described with its true longitude $L$, the objective is to benefit as much as possible of a reduction of the heliocentric velocity norm that's why the planet used is Vulcan for its biggest $mu$
and the true longitude evolve from %frac{/pi}/{2}$ to  %frac{3 */pi}/{4}$
		- The initial velocity Vx varied from 5 (km/s) to 50 (km/s)

For each combinations of those parameter we look for the Y and Z coordinates in the heliocentric frame at the problem initialization that satisfies the planet position constraint.

We use the Nelder-Mead implementation from scipy and a bissection method on the time of flight to find the closest appproach (min euclidean distance) point from the initial orbit to the planet's position.


2. Build two legs : Once the table is complete, the two first legs are computed as follow.

'We shuffle the planets list.
'We iterate over Vulcan complete revolutions until we find a solution   
	'We iterate over the planets list :
		We perform a bounded minimization over the following variables 
		##minimize pb parameters are [tof (second leg), V0 (s/c velocity at mission beginning), L (orbital posiiton of planet Vulcan at encounter)]##
		bounds = [
			(1.0, 86400 * 365.25 * 10),
			(5000, 50000),
			(1.745329, 2.268928),
		]
		
		The second and third variables are used within the minimizer to interpolate the table built at step 1, output from the interpolator are the state of the S/C at Vulcan encounter and the absolute time from t=0 (mission begin).
		The obtained variables are used to solve Lambert problem between the arrival planet (current planet of the iterated list) and the necessary DV at departure is computed.
		The minimization algorithm used is L-BFGS-B with the DV as criterion, again we use the scipy implementation.

		From here we iterate until a DV relatively small is obtained; 100 m/s has been chosen. 
		
		Then another minimization is performed to refine the solution, this time not relying on the table, 
		the variables are : 
		##Parameters are Y0, Z0, V0, L(Vulcan true longitude at encounter), tof(flight time from planet 1 to planet 2)##
		and the criterion is built as follow : normalized euclidian distance at Vulcan closest approach + normalized DV needed at Vulcan departure to acheive the second leg, methos is L-BFGS-B (not that tof is unbounded so a specific treatment is applied to ensure positiveness of the tof). 
		
		Once tolerance is acheived we exit this loop to step 3.
		
		Note that up to now only conic arcs are allowed (the solar sail is not used).
		

- The two first steps have been designed this way to acheive the following objective : Maximize the heliocentric velocit reduction of the S/C allowing high incoming velocities in order to not consume too much mission time on the Altaira system approach.
- Build a two steps approach with a table to greatly ease convergence. 

3. Orbital period reduction : 

- At this point we have noted that jumping directly to the scoring algorithm resulted in poor solution and was not adapted to maximize the score in presence of highly elliptical (and long period) orbits at the end of the first leg. 
- It was then important for many cases to ensure to obtain at least another leg that would decrease the orbit energy before completing another completer revolution of the S/C orbit. 

- We then use the same algorithm that will be used later to maximize the score but with a different score computation which is : maximal reduction of the  S/C orbital period. 

- This algorithm is exited as soon as the target max orbital period threshold is acheived, the treshold is set to 10 years.

The algorithm is as follow: 
'iterate over the lists of all bodies (comets, planets, asteroids) 
	iterate over the range of time of flights between 10 days to 50 years with a 100 steps discretization ( the max tof is bounded by the mission maximal duration)

		Solve the multi-revolutions lambert problems (clock-wise and counter clock-wise) up to 100 revolutions between the previous planet to the current planet on the list.
		compute the DV necessary to perform the leg 
		
		if  DV < 200 m/s
			A minimization algorithm with the DV as criterion and tof as a parameter is performed.
			
		if the DV is below tolerance the arc is validated.
		
		otherwise if the only conics option is selected the algorithm jump to the next tof
		if solar sail arcs are allowed a filter is called to quickly identify if the arc is solvable with the sail or not, if solvable the arc is validated.

		The score is then computed and stored
		
At the end of the loop we keep the flyby with the maximum score.



4. Increase the score 

- The same algorithm is applied but the score is simply computed as : $GTOC score increase / tof_leg$ 
- The algorithm stops when no next leg is found;

5. Solving of the solar sail arcs

	- TBD 
	- in case of a failure in the arc solving we take the next solution from the algorithm above and again until convergence is acheived. 


Note : 
   Although not stated here constraint like min Altaira distance, min and max flyby altitudes are embeded within the minimization methods. 
   When using only conics arc, the asteroids and comets + Yandi are removed as those solution often led to no next leg.
		


This methodology is then repeated N_times as the randomness in the process always lead to differents sequence and so different scores. 
The algorithm has been able to obtain a score of around 70 points and score over 40 points are usually obtained within 10 iterations 
Some or those case are stored in the results folder of this repo. 


