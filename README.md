# Truly understanding Machine Learning
During my time studying Data Science and Machine Learning, Software Development, Computer Science, Physics, and Mechanical Engineering, I have learned a lot about the best way the I learn. It is clear at this point that for a beginner, jumping right into a text book is rarely the best route to follow. The concepts, and more importantly just the language and terminology, will seem very difficult to comprehend, and most likely leave you discouraged that the material is simply outside of your grasp.

This is especially true in the field of Data Science and Machine Learning, where several other technical disciplines intertwine:
- Statistics
- Probability
- Computer Science
- Calculus
- Linear Algebra
- Information Theory

By jumping into formula heavy text books, and skipping out on the real world applications of Machine Learning (which lets be honest, they are pretty awesome), there is no incentive to continue and push forward. This is where the Top down approach was introduced. School systems generally teach via a bottom up approach - giving students the small building block thats can be combined in the end to create a grand system. Again, this leaves the learner wanting more, and often struggling to connect the dots of *why this small building is useful* and *why should I care*?

The top down approach throws you right into the deep end, allowing you to work with premade algorithms and libraries, without fully understanding the math and intuitions behind the overall system. This, in my opinion, is a much better approach, but still leaves a bit to be desired. As with anything, I feel that balance is the key here. The goal should be to use real world examples to teach the mechanics of what is going on under the hood. To often concepts that are taught are treated as black boxes, and rote memorization is used to get through. This worked for a time, but in the field of Data Science and Machine Learning, it will not. There is no one size fits all - it is messy, chaotic, and unclear. And that is **my job** as a Data Scientist - to bring clarity to a problem, and help find a resolution.

The goal of this blog is to build **intuitions**. Anyone can follow a basic process of predetermined steps and arrive at a solution. But we want to create intuitions of what is *actually* going on, so that if the situation was broken from its cookie cutter form we would be able to take that in stride and still make sense of based on the fundamental principles. So with that said...

This blog (built from my [intuitiveml notebooks](https://github.com/NathanielDake/intuitiveml)) is designed with two main purposes in mind:
1. Highlight my exact journey of teaching myself Machine Learning and Data Science
2. Develop key intuitions about what is really happening. 

<br>
<br>

**Content**<br>
The general content of this blog is arranged as follows:
* **Deep Learning** 
<br> This section contains everything from the fundamentals of FeedForward Networks and Vanilla Backprop, to Modern Deep learning techniques such as adaptive learning rates and batch normalization, and finally Recurrent Neural Networks and their application to Natural Language Processing problems. Convolutional Neural Networks coming soon. 

* **Artificial Intelligence**
<br> Focusing mainly on Reinforcement learning, specfically the Explore Exploit Dilemma (and bayesian techniques), Markov Decision Processes, Dynamic Programming, Temporal Difference Learning, Q-Learning, and Approximation Methods. Reinforcement learning with deep learning techniques coming soon. 

* **Machine Learning**
<br> This section is the most broad by far consisting of Linear Regression, Logistic Regression, Decision Trees, Probabilistic Graphical Models, Ensemble Methods, Unsupervised Learning, and Hidden Markov Models. 

* **Natural Language Processing**
<br> Currently this section is empty, however, I do have the first batch of notebooks available on my github that deal with semantic and latent semantic analysis. This section is one that will be heavily expanded on, specifically applying deep learning techniques (such as recurrent neural networks) to solving NLP problems. There is also the matter of production level NLP pipelines which I have dealt with during my time working for Carimus, which will be added when I have the time.

**An Individual Notebook**<br>
I should mention that each individual notebook (which is built from a jupyter notebook) contains what I personally feel is crucial to understand a given topic. Code samples are always written via *python*, and are mixed with visualizations (made by me), equations, pseudocode, and whatever else is needed to ensure a clear and effective transfer of knowledge. 

With that said, here is one final quote to always remember (and one that I remind myself of as I put together each one of these notebooks). It is from the renound American Physicist Richard Feynman as he was attempting to explain Fermi-Dirac statistics:

> Feynman was a truly great teacher. He prided himself on being able to devise ways to explain even the most profound ideas to beginning students. Once, I said to him, “Dick, explain to me, so that I can understand it, why spin one-half particles obey Fermi-Dirac statistics.” Sizing up his audience perfectly, Feynman said, “I’ll prepare a freshman lecture on it.” But he came back a few days later to say, “I couldn’t do it. I couldn’t reduce it to the freshman level. That means we don’t really understand it.”

# How is this repo setup
This repo consists of several main directories
1. Deep Learning
2. Machine Learning
3. Artificial Intelligence
4. Mathematics
5. Notebooks

All of the notebooks in directories 1-4 have been converted into blog posts and can be seen on [my personal site](www.nathanieldake.com). I highly recommend this for a better viewing experience (typography and formatting was chosen very carefully to make for the best viewing experience possible). However, as always, working with the code directly is absolutely invaluable, so I still would have the jupyter notebook kernal running the notebook as you view the post right along side it.

Directory 5, `notebooks`, contains all notebooks that are _complete_ from a content standpoint, but not quite ready from the typographical/presentation side of things to be live on my blog. This contains a ton of content on Linear and Logistic Regression, Bayesian Techniques/Probabilistic Graphical Models, Decision Trees, Ensemble Methods, KNN, NLP, and several others.\

# Setup Instructions
1. Navigate to the directory where you would like to be storing this repo
2. Run `git clone https://github.com/NathanielDake/intuitiveml.git` 
3. Change directories into that notebook, i.e. `cd intuitiveml`
4. Run `jupyter notebook`
5. This will spin up the notebook frontend as well as the kernel. 
6. Stay up to date with the most recent changes by running `git pull`

# Notes
1. All of the code has been written in `python 3`, so you will want to ensure that you have `python 3` installed successfully on your machine in order to run the notebooks in this directory. 

# Good Luck!
I wish anyone following along with these notebooks the best of luck on the Journey to understanding Machine Learning and Data Science.
