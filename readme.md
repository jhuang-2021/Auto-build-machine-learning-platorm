# MLApp : an end-to-end API for training machine-learning models

# Purpose

This project implements an end-to-end GUI for training machine-learning models.
The GUI is based on Python 3.7, flask framework, HTML and CSS.

The SQLAlchemy package is used to manage the client machine-learning applications.

# Implementation description

The training data sets are imported by uploading the clients' csv files
into the server. 

In the back-end, a set of 5 machine-learning classification models are implemented,
all of which are based on scikit-learn package.

The clients can train and test the model by click on a HTML button,
and get back the test report from the web-page. 

This interface can be easily extended to include any kind of machine-learning
models. 

# Usage

To install the package, Python3 and pip are needed in your system. I
tested every functionality on a Linux virtual machine, Ubuntu 18.04

1) Download the package
2) unzip mlApp.zip
3) cd mlApp
4) pip install -r requirements.txt
5) python api.py 

In your localhost, at the URL address text input area, type:

   http://0.0.0.0:5000/

You will see the interface.

# Some notes

For simplification, the development server associated with flask is used.
This is not an efficient server. For production, I suggest that you should use
gunicorn server, which supports multi-processes. 

If you have any issues, please send email to:

jian_bo_huang@outlook.com



