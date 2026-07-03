# Contribution Guidelines
Thanks for considering a contribution to EventCV! We are still in the process of formalizing processes like contribution which will naturally evolve and change. Below is the current process we would like you to follow.

## Contribution workflow
1) [Create a fork](https://github.com/EventLAB-Team/eventcv/fork) of the EventCV repository from `main`.
2) Add or modify the code files you wish to contribute, if adding new functionality please ensure a test is included under `tests/` for code coverage analysis.
    - For details on what a good test looks like, please see the existing tests
3) [Create a pull request](https://github.com/EventLAB-Team/eventcv/pulls) and ping `@Event-LAB-Team` or a specific user to review
4) Once reviewed and validated, your changes will be merged into `main` and scheduled in a new release of EventCV

## FAQs
- Can I use an AI agent to code changes or add features?
    - Yes, like any modern workflow AI coding agents are extremely useful and can be used. However, all code will be reviewed by a human (the EventLAB-Team, and ideally yourself) and any poor quality pull requests will simply be closed. 
- If I contribute a rust crate, do I need to provide Python bindings?
    - Yes, you must provide the necessary Python bindings through PYO3 and maturin if contributing a new feature written in rust. 