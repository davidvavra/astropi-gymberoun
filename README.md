# astropi-gymberoun

AstroPi project for team "Barrande" from Gymnázium Joachima Barranda, Beroun, Czechia, 2022/2023

## The experiment

This is a Life on Earth experiment, analyzing images of Earth from ISS.

### The goal

The goal of the experiment is to test feasibility of running machine learning topography classification directly on the
device with limited resources (such as in a cubesat). We plan to process images also afterwards on Earth and compare the
results.

### How does it work?

- Our program has a 30s loop and runs for 3 hours
- We take picture of Earth and crop it to a square with a circular mask (like the window on ISS)
- We run machine learning classification of the topography directly using Coral ML chip. We trained the model on Earth
  using images from previous years and manual classification as source. The output is a masked image with a different
  color for different topography features. We also save this coverage in percents to a CSV file.
- We also collect data from all available sensors for the purposes of secondary mission. We also get ISS location and
  save everything to a CSV file.

### Authors
- Cyril Šebek: machine learning code
- Jonáš Koller: pre-processing images code
- Filip Rosický: sensor code
- Matyáš Leon Dušek: connecting everything together, testing, error handling, logging
- Adam Vášek: ISS location code, CSV handling
- Adam Denko: secondary missions
- David Vávra: mentor, project management, final touches and testing