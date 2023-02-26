# Airline-Passenger-Satisfaction

<br><br>
## Files here arranged :
1 - Airline-Passenger-data-notebook.ipynb <br><br>
2 - pipelines_description.txt <br><br>
3 - Pipelines.py <br><br>
4 - Pipeline-Selection.ipynb <br><br>
5 - Model-Building.ipynb <br><br>

<br><br>
## Let's go through them and see what do the do...

### First Airline-Passenger-data
#### In this notebook I analyze the data and understand what feature engineering and data preprocessing techniques to use
> - I understood the data.<br><br>
> - I built dummy models.<br><br>
> - I stated what preprocessing techniques to use.<br><br>

<br><br>
### Then I made description for my pipelines
#### Where I wanted to do 4 pipelines and I stated what will they do in this txt file and there names :
> - Pipeline 1 <br><br>
> - Pipeline 2 <br><br>
> - Pipeline 3 <br><br>
> - Pipeline 4 <br><br>

<br><br>
### Then a python file containing the code for this pipelines and there is :
> - custom classes that inhert from sklearn to do custom cleaning <br><br>
> - 4 pipelines functions <br><br>
> - compute_pipeline function which summarize the performance of the pipeline (will be called in the next notebook) <br><br>

<br><br>
### And Pipeline Selection Notebook 
#### which runs these pipelines through data and see which one gives good performance
> - It summarize every Pipeline<br><br>
> - gives table for avg models for each pipeline<br><br>

<br><br>
### Model Building
#### Here we noticed that ensamble learnig models performs very well on this data so :
> - I traind many ensamble learning classifiers (Randomforest, ExtraTrees, GradientBoasting, AdaBoast, Bagging classifier, voting classifier)<br><br>
> - I noticed that the best to use is `Random forest` or `LGBM Classifier`<br><br>
> - Knowing the bias and variance... I Hyper tuned Random Forest and now the tuned Random Forest is the best model<br><br>
> - Then I built a small Neural Network and although it didn't take much effort but it was very promising

<br><br>

## What to do next ?
> - I will try to reduce variance in my random forest classifier
> - I will try to develop the Neural network to make it give better preformance
