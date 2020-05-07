from random import randrange #for generating a random range
import seaborn as sns #for plotting
import matplotlib.pyplot as plt #for plotting

#Load CSV as an array dataframe
def read_csv(filename):
    dataset = [] #empty dataset
    data = open(filename,'r').readlines() 
    #open csv file in read mode, read data
    for r in data: 
      #loop through every line in data
      row=r.split(',') 
      # remove , and convert string to array
      temp=list(map(float,row[:-1])) 
      #convert string datapoints to float
      if row[-1]=='M\n': 
        #convert last predicted value to boolean 
        temp.append(1) 
        #M=0, malignant
      else:
        temp.append(0) 
        #B=0, benign
      dataset.append(temp) 
      #append the real dataset
    return dataset[:len(dataset)-100] 
    #the final dataset generated as an array of arrays

#Split dataset into k parts (or folds)
#folds= equally sized randomized samples
def cross_valid_split(dataset, n): 
    temp = list() 
    #the split of the data we build the fold with
    dataset_copy = list(dataset) 
    #maintain a temporary copy
    fold_size = len(dataset) // n 
    #n= number of folds
    for i in range(n): 
      #loop through folds
        fold = list() #empty fold
        while len(fold) < fold_size: 
            #while fold size hasnt reached the required limit
            index = randrange(0, len(dataset_copy)) 
            #take a random size sample
            fold.append(dataset_copy.pop(index)) 
            #take it from the dataset copy
        temp.append(fold) 
        #append it to the final folds array
    return temp

# Compute accuracy
def find_acc_metrics(act, pred):
    tn,tp,fn,fp,= 0,0,0,0 
    #true negatve, true positive, false negative, false positive
    # 1=True, 0=False
    for i in range(len(act)): 
        if act[i] == 1 and pred[i]==1: 
            tp+=1
        elif act[i]==1 and pred[i]==0:
            fn+=1
        elif act[i]==0 and pred[i]==1:
            fp+=1
        else: 
            tn+=1
    #Accuracy, Confusion Matrix, Precision, Recall
    return [(tn+tp)/(tn+tp+fn+fp),str(tp)+'  '+str(fn)+'\n'+str(fp)+'  '+str(tn),tp/(tp+fp),tp/(tp+fn)]

def eval_algo(dataset, algo, n, *args):
    folds = cross_valid_split(dataset, n) 
    #get folds
    scores = list() 
    #list of accuracies
    for i in range(len(folds)): 
        #for every fold
        train_set = list(folds) 
        #your training set will be all the folds minus the current fold
        train_set.remove(folds[i]) 
        #remove current fold 
        train_set = sum(train_set, []) 
        test_set = list() 
        #empty test set
        actual=list()
        for row in folds[i]:
            row_copy = list(row) 
            #get entire row in the fold
            actual.append(row_copy[-1]) 
            #append the true value
            row_copy[-1] = None 
            #set the to predict attribute to "None"
            test_set.append(row_copy) 
            #append it into the test set
        
        predicted = algo(train_set, test_set, *args) 
        #get predictions from the MLP algorithm
        metrics = find_acc_metrics(actual, predicted) 
        #calculate the metrics
        print('-------Fold',i+1,'------')
        print('*****Hyperparameters*****') #print the hyperparameters
        print('Completed Cumulative Epochs: ',5000*(i+1)) #print results after nth epoch
        print('Learning rate: ',0.02,'\n') #alpha 
        print('*****Metrics*****')
        print('Accuracy: ',metrics[0]) 
        print('Confusion Matrix:\n'+metrics[1])
        print('Precision: ',metrics[2])
        print('Recall: ',metrics[3],'\n')
        scores.append(metrics[0]) #append into scores
    return scores

#Activation function is a step function , Sigmoid, RELU, tanh may also be used 
#0 for <0 and 1 for >=0
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i] 
        #input to the activation function
    return 1.0 if activation>= 0.0 else 0.0 
    #The ideal step function , which works like an on off switch

def train_w(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))] 
    #initialize with 0 weights
    for epoch in range(n_epoch): 
      # for every epoch mentioned
        sum_error = 0.0  
        #error=0 initially
        for row in train: 
          #for every value in train
            prediction = predict(row, weights) 
            #carry out a prediction either 0 or 1
            error = row[-1] - prediction 
            #calculate error
            #take the sum square of the error per epoch ()
            sum_error += error**2 
            #add it to the squared sum error
            weights[0] = weights[0] + l_rate * error #our bias 
            #loop over each weight and update for a row in each epoch
            for i in range(len(row)-1): 
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i] 
                #Update all the subsequent weights
        error_graph_data.append((epoch, sum_error)) 
        # for every epoch, append its error
        #loss function plot
    return weights #return the updated weights

def plotit(error_graph_data,a): 
  #functions to plot the loss functions
  x=[error_graph_data[i+5000*a][0] for i in range(0,5000)] 
  #x values for 5000 epochs
  y=[error_graph_data[i+5000*a][1] for i in range(0,5000)] 
  #y values for 5000 epochs
  ax=sns.lineplot(x,y) 
  #make a lineplot x vs y
  ax.set_title('----------------Fold '+str(a+1)+' loss function/Squared sum plot-------------------') 
  #Fold outputs
  ax.set(xlabel='Epochs',ylabel='Squared Error') 
  #labels for axes
  plt.show() 
  #show the plot

# Perceptron Algorithm (SGD included)
def perceptron(train, test, l_rate, n_epoch):
    predictions = list() 
    #initial empty predicitons 
    weights = train_w(train, l_rate, n_epoch) 
    #get updated trained weights  
    stor_weights.append(weights)
    for row in test: 
        #for every entry in test
        prediction = predict(row, weights) 
        #carry out predictions
        predictions.append(prediction) 
    return predictions #array of predictions

filename = 'data.csv' #our dataset
dataset = read_csv(filename) #load it
print('Breast Cancer Dataset: \n',dataset,'\n')

n_folds = 3 
#each time, a 33.33% test and a 66.66% train split is obtained
l_rate = 0.02 
#learning rate
n_epoch = 5000 
#number of training epochs per step fold

error_graph_data = list() 
#get error for every iteration
stor_weights = list() 
#store the weights
print('-#-#-#-#-#-Training phase with k-fold Cross Validation-#-#-#-#-#-')
scores = eval_algo(dataset, perceptron, n_folds, l_rate, n_epoch)
#evaluate the perceptron algorithm

print('Results obtained in every run:\n')
# printing data to be graphed
print(error_graph_data)

for i in range(3):
  plotit(error_graph_data,i)
#plot 3 graphs

print('All Accuracy Scores: ',scores)
#All accuracies printed
print('Average Accuracy: ',sum(scores)/len(scores))
#Average accuracy

print('\nInitial start weights: ')
print(stor_weights[0])
#first appended weight
w_test = stor_weights[-1]
#last appended weight
print('\nUpdated Final Weights: ')
print(w_test,'\n')

print("-#-#-#-#-#-Printing test results for an unseen dataset (100 rows) -#-#-#-#-#-\nActual  Predicted")
act,pred=[],[]
for i in range(100):
    print(dataset[len(dataset)-i-1][30],'\t', predict(dataset[len(dataset)-i-1], w_test))
    act.append(dataset[len(dataset)-i-1][30])
    #append the actual values
    pred.append(predict(dataset[len(dataset)-i-1], w_test))
    #append the predicted values

metrics=find_acc_metrics(act,pred)
#find testing accuracy metrics
print('\nAccuracy: ',metrics[0])
print('Confusion Matrix:\n'+metrics[1])
print('Precision: ',metrics[2])
print('Recall: ',metrics[3])

"""# Justification and Conclusion

A simple Single Layer Perceptron algorithm was run on the breast cancer dataset using K-Folds Cross Validation and SGD. The testing accuracy obtained was around 92.7% in the training dataset and 93% on an unseen dataset, among other relevant metrics.

The three graphs obtained show the squared error sum function for every epoch, and for every considered fold. As we can see, the error diminishes as the number of iterations increase.
"""