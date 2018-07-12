# keras-regression
A neural network for regression problem

-------------------------
The Task
The task is to create a neural network which takes a set of 10 points as inputs, and outputs slope and the y-intercept of the best-fitting line for the given points. The points are noisy, i.e. they won't fit perfectly on a line, so the net must figure out the best-fit line.

-------------------------

-------------------------
Inputs

The input is a CSV file consisting of the set of 10 points that need to be fit, along with an id for each row. For example:
id,x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9
0,15.104372521694373,-68.3288669890148,-95.58279088853587,-181.47566659777777,39.29508013857384,-34.02032092432305,89.77504252047551,13.335549409789905,59.18020460838781,-27.745988520228575,58.18208066946731,-20.139440271026118,114.15065113682351,22.885664222726238,54.91953213577344,-40.08393846757413,96.94956199134829,-1.7529553733813685,43.73834621553149,-48.32482436599675

-------------------------

-------------------------
Ground Truth

The ground truth format gives the desired slope and intercept for each row. For example, the truth corresponding to the above rows is:

id,slope,intercept
0,0.9999146924638005,-84.03411330249395

-------------------------

-------------------------
Evaluation
The evaluation wil be on two metrics: the mean squared errors of the slopes, and the mean absolute errors of the intercepts. The attached evaluate.py file will take two ground-truth-format files and print out the evaluation.

To give an example of the expected results, the best net we got had the following evaluation on the data it was trained with:


$ python scripts/evaluate.py data/train_100k.truth.csv env/submission.train_100k.csv
Slope mse: 0.00682871657263
Intercept mae: 4.60931900967

-------------------------

-------------------------
Approach

It was made a neural network fully conected with 6 layers, the input layer has 50 neurons followed by 4 hidden layaers with 40, 30, 20, 10 neurons, then the output layer has 2 neurons. For the input and hidden layers it was used the relu activation function and for the output layer it was used the linear activation funcion. The optimizer was the rmsprop and we use 300 ephocs.

Fine, but how do we got there...?

First we built a simple 3 layers neural network with 20, 10, and 2 neurons each layer, we measure the mse and mae for this configuration and compare with other variants of this network changing the number of neurons in each layer and its activation funcion. For each configuration we separate the database in training and testing and evaluate its results. For this round we saw that some activation funcions didn't work well for this problem, and at this level the number of neurons in each layer doesn't show as much impact on the mse as we wanted. Thus, in this round we could pick the best arrange of activation function, however the outcome wasn't yet as good as we wanted.

Second, we tried to do some pre processing on the database using the NN from the previous round. The first pre processing method we used was to normalize the data, we took the mean and standard desviation from each of the 20 attributes on the entire database and normalize them. As we study the database we realize that each sample (x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9) could be processed as a small database of 10 two-dimensional points, so we tried to normalize each sample alone taking the mean and standard desviation from the 'x' and 'y' coordinates of its 10 two-dimensional points. We also project the database using PCA calculated from the entire database and from each sample. The last pre processing method we tried to use was to grouping the 10 two-dimensional points from each sample in fewer points hoping to decrease the influence of outliers, the grouping process was made using both mean and median. At the end, none of the pre processing methods used here helped to decrease the mse and the mae on the experiments, thus we decided to use no pre processing method.

Finally, the last thing we tried was to increase the number of layers on the neural network. We raise it from 3 layers to 6  testing a bunch of different configurations between them. Besides the number of layers, we also tested the number of neurons in each layer trying to find the best configuration. At the end we saw that as deeper(more layers) we go as good the results were, and now the number of neurons in each layer has a stronger influence on the final result. 

So... that's how we got the final neural network we present here. 

A important thing to say is that from the experiments we could see that if we had going deeper (add more layers) on the network we believe that we the results coud be better, however, the limitations of hardware could not allow it.

-------------------------

-------------------------
Running the code

To run this code you need to pass it 4 arguments, the path of the training data, the path of label data, the path of test data, the path were you want to save the test predictions.

The code bellow is a exemplo of how to run it:
$ python src/main.py database/train/train_100k.csv database/train/train_100k.truth.csv database/test/test_100k.csv submission.csv

the code above execute the main file found at src/main.py giving it the training data found at database/train/train_100k.csv, the training labels found at database/train/train_100k.truth.csv, the testing data found at database/test/test_100k.csv, and store its prediction on the testing data at the current directory with the name submission.csv

-------------------------

-------------------------
Submission file

The prediction on the test data can be found at the diretory results/submission_test.csv
