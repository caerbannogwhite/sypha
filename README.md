# README #

### Model ###

Model should be an abstract class to collect the features of a mathematical model. A Model instance
should be returned, for instance, when reading an LP file or and SCP file.
Possible implementations:
* ModelLP
* ModelMILP

Main features:
* Variables: a list of Variable objects (type, name, objective)
* Constrains: a list of Constraint objects (row, name, sense, rhs)

### Node ###

...

A Node object maintains the information to get its sub-model representation (crash procedure) from
the original model.



This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact