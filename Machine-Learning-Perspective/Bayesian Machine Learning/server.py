# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
from __future__ import print_function, division
# from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from flask import Flask, jsonify, request
from scipy.stats import beta

# create an app
app = Flask(__name__)


# define advertisement 
# there's no "pull arm" here
# since that's technically now the user/client
class Advertisement:
  # initialize instance clicks and views 
  def __init__(self, name):
    self.name = name
    self.clicks = 0 
    self.views = 0

  # method to draw a random sample from current class instance beta distribution
  # to start, Beta(1, 1) is the prior
  def sample(self):
    return np.random.beta(1 + self.clicks, 1 + self.views - self.clicks)

  # add 1 to class instance click attribute (this is used to calculate beta dist and in sampling)
  def add_click(self):
    self.clicks += 1

  # add 0 to class instance click attribute (this is used to calculate beta dist and in sampling)
  def add_view(self):
    self.views += 1

# initialize our advertisements
advertismentA = Advertisement('A')
advertismentB = Advertisement('B')


# in the bayesian algorithm, getting an add is parallel to determine which arm to draw in a 
# multi armed bandit scenario. In this case, we want to sample from the beta distribution 
# associated with advertisment A and advertisment B, return the advertisment whose 
# sample was higher
@app.route('/get_ad')
def get_ad():
  sample_a = advertismentA.sample() 
  sample_b = advertismentB.sample()

  if sample_a > sample_b:
    ad = 'A'
    advertismentA.add_view()
  else :
    ad = 'B'
    advertismentB.add_view()
  return jsonify({'advertisement_id': ad })

# in the bayesian algorithm, clicking an add is parallel to pulling the bandit arm and winning
# so in this case the user was shown an add (pulling the handle of a bandit arm), and then 
# clicked the shown ad (the bandit arm paying out). So what needs to be done is the Advertisement
# instance number of clicks attribute needs to be updated. If the user does not click, 
# nothing needs to happen, since the number of views was already updated in the `get_ad` 
# method

@app.route('/click_ad', methods=['POST'])
def click_ad():
  result = 'OK'
  if request.form['advertisement_id'] == 'A':
    advertismentA.add_click()
  elif request.form['advertisement_id'] == 'B':
    advertismentB.add_click()
  else:
    result = 'Invalid Input.'

  # nothing to return really
  return jsonify({'result': result})


if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8888)