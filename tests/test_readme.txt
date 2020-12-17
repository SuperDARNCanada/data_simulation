The simulator is tested by seeing if it can create a statistically accurate recreation of real data.
To do this, a fitacf record was chosen to build simulation parameters from. The corresponding IQ
samples were also extracted in order to compare to the output of the simulation. To simplify
future testing, testing inputs were extracted and stored into a separate file.

In order to verify the simulation, 100 simulations of the averaging perioud are done using the
parameters created from the fitacf record. The resulting samples are overplotted on top of each
other in low alpha to create a window for which the real samples should lie in. The real samples
are then plotted on top to see whether they lay within the window and whether the amplitudes
match. Finally, vertical bars are plotted to show the location of TX blanked samples in the real
data.

The second plot shows a histogram of the simulated samples with a histogram of the real samples
overplotted in order to show that both should be Gaussian distributed.

Future work can use the test input data and compare against the plot/output data.
