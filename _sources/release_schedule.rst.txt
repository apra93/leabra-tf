================
Release Schedule
================

This is a general schedule of tasks to be completed for the project to reach
the various release milestones.

0.0 Release
-----------

- Create a documentation page that will automatically add new notebooks as they
  are created.
- Create a new README that details how to go about the repo, and move components
  (like this release schedule) to their own pages in the documentation.
- Implement the task described in O'Reilly's paper.
- Establish a baseline using other canonical deep networks (DNs).
- Establish a baseline on the task using ``leabra7``.
- Compare against results shown in O'Reilly's paper.
- Run the same DNs on TPUs, and establish baselines between GPU and
  TPU performance.
- Pull out competitive inhibition learning component from leabra and add it to
  the canonical DNs.
- Pull out hebbian learning component from leabra and add it to the canonical
  DNs.
- Compare modified DNs (MDNs) to results shown in the O'Reily paper (perhaps
  strip leabra of each component to make clear comparisons).
- Find a way to reference other notebooks (analyses) like in a paper.
- Recreate results in TPU if they weren't already.
- Implement competitive inhibition and hebbian learning in DNs in GPUs and TPUs.
- Assess performance compared to leabra.
- Mass-parameter search on the TPUs to compare with leabra.
- Revaluate different versioning packages and write down reasoning for the
  choice.

1.0 Release
-----------

- Streamline analysis pipeline to be reproducible on your own machine (if not
  already)
- Write implementation tests (should have some if not all already).
- Write analysis tests from notebooks.
- Look into running unit tests on a local machine as analysis testing will get
  computationally expensive.
- Set up continuous integration (CI) either locally (ex. jenkins) or on the
  cloud (ex travisci, circleci).
- Set up repo coverage.
- Compile notebooks into a 1.0 release report.
- Write documentation, add references, and fill out READMEs in various parts of
  the repo.
- Setup vulture and assess how well it can prune repos
- Prune away by hand anything vulture misses.
- Make all tests (implementation and analysis) pass with the pruned repo.
- Make pull-request (PR) and wait three days before merging. Don't work on it in
  between.

1.1 Release
-----------

- Revisit containerization of packages and choose one to go forward with and
  document reasoning.
- Containerize the package and ensure tests pass on all relevant machines.

1.2-1.5 Releases
----------------

- Establish baselines of state of the art DNs (SDNs) on MNIST (or another task)
  using TPUs.
- Establish baselines of MDNs on this same task using by hand on TPUs.
- Mass parameter search for optimal MDNs on the TPUs.
- Recreate relevant portions of 1.0 analysis on MNIST and compare the results.

1.6-1.9 Releases
----------------

- Recreate full leabra in tensorflow, then compare against SDNs and MDNs.

2.0 Release
-----------

- Streamline analysis pipeline to be reproducible on your own machine (if not
  already)
- Write new implementation and analysis tests, then add to CI pipeline.
- Compile notebooks into a 2.0 release report, and write documentation.
- Make PR and wait three days before merging. Don't work on it in between.

2.1 Release
-----------

- Look into `research objects <http://www.researchobject.org/>`_, and assess
  the difficulty of getting it in that form.

