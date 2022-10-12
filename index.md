---
layout: page
title: Next-Generation Precision for Neutrino and Collider
Computations
subtitle: A SciDAC-5 Project
---
The connection between particle physics theory and experiment hinges on the simulation of
detector events by means of Monte-Carlo methods. Event generators are
central to the success of the accelerator-neutrino program, which will
provide stringent tests of the three-flavor paradigm. A quantitatively
correct modeling of neutrino interactions entails multiple energy 
scales and becomes computationally hard when including a fully quantum
mechanical treatment of nuclear many-body effects, as may be needed to
reach the precision targets at the Deep 
Underground Neutrino Experiment (DUNE). Event generators are also crucial components of
the Large Hadron Collider (LHC) software stack. They can nowadays perform parton level
simulations at high precision, which are subsequently matched to parton showers to provide
particle-level events. With the upcoming data increase at the LHC, the computational footprint
of precision event generators will grow dramatically, contributing to the outpaced growth in
computing needs compared to available budgets. Event generation is a non-negligible fraction
of LHC computing, and improved performance will help to reduce
computational requirements.

This project aims to address the computing problem by constructing a performance portable
solution that optimizes basic math operations needed in parton- and particle-level event gen-
erators for a variety of architectures, parallelizes calculations across events, and capitalizes on
heterogeneous systems. First applications will be a novel theory-driven event generator for
the DUNE experiment, and the Monte Carlo for FeMtobarn processes (MCFM) for the LHC.
This will serve as a blueprint for particle-level event generators and automated matrix-element
generators. Guided by the known matching and merging algorithms in collider physics, the
project will explore various possibilities to extend the current classical cascade models in neu-
trino event generators to a partial quantum cascade, for which computational efficiency is an
essential prerequisite.


<h2>Events </h2>

#Core-collapse Supernova and Gravity Effects on Pool Boiling.

#<div class="custom-row">
# <div class="custom-boiling">
 # <img src="{{site.baseurl | prepend: site.url}}/assets/img/d96_3d3d_after.jpg" style="width:100%">
# </div>
# <div class="custom-boiling">
 # <img src="{{site.baseurl | prepend: site.url}}/assets/img/boiling_lowg.gif" style="width:100%">
# </div>
# </div>

                                                                  
                                                                  
# Code of Conduct

All collaborators are expected to be respectful of one another. Discussions and disagreements should remain polite and free of abusive language.

There will no discrimination based on gender, ethnicity, race, sexual orientation, or any other potentially divisive reason. 

All major code components will have an associated list of authors.

Proper credit must be given to the contributors of code components used in all scientific publications.

All code contributors will follow the contribution policies and workflow and the coding standards.

