# Astra_robotics_life_science_model
A machine learning model for the URC Rover Challenge 

----------------DATASET EXPLANATION-------------------
1. The dataset used for this model is a synthetic one, which we generated through the website gretel.ai, the dataset has around 250 records of input columns.
2. The input parameters are: humidity, soil depth, temperature, organic content, ph, soil moisture, carbon content, nitrogen content, oxygen content, sulphur content, and the output parameter is the probability of presence of life.
3. In the code, the csv files "Labelled2.csv" and "Unlabelled.csv" represent two things, which is a little counter-intuitive:  "Labelled2" has no output parameter and is used only for testing if the model runs; "Unlablled" has the ouput parameter as well, and is used to train the model
-------------------------------------------------------

-----------------COMMON CODE FOR BOTH THE MODELS-----------------

        1.pip install scikit-learn
        2.import pandas as pd
        3.from google.colab import files
          uploaded=files.upload()
        4.import io
          df1=pd.read_csv(io.BytesIO(uploaded['Labelled2.csv']))
          df2=pd.read_csv(io.BytesIO(uploaded['Unlabelled.csv']))
These lines are same for both the codes, they perform the following actions:
1. This line installs the sci-kit library of python
2. This line imports pandas library of python, and uses it as pd
3. These lines are used to upload our csv files from the local system, i.e "Labelled2.csv" and "Unlabelled.csv"
4. These lines are used to create the dataframes for the two datasets, which will be used throughout the code
--------------------------------------------------------------------

--------------------K-NEAREST NEIGHBOURS-----------------------------
   1. import pandas as pd
      from sklearn.model_selection import train_test_split
      from sklearn.neighbors import KNeighborsRegressor
      from sklearn.metrics import mean_squared_error
      from sklearn.preprocessing import StandardScaler

    2.# Assuming X is the input features and the last column is the target variable
      X = df2.iloc[:, :-1]  # Assuming last column is the target variable
      y = df2.iloc[:, -1]
      
    3.# Splitting the data into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      
    4.# Standardize the features (mean=0 and variance=1)
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)
      
     5.# Creation and training the MLP model
      knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors (k) here
      knn_model.fit(X_train_scaled, y_train)
      
      # Make predictions on the test set
      y_pred = knn_model.predict(X_test_scaled)
      
      # Evaluate the model
      mse = mean_squared_error(y_test, y_pred)
      print(f'Mean Squared Error: {mse}')
These lines of code are the training code for the model: 
1. importing all the necessary libraries for the training
2. We are spilting the columns of the dataset into X and y, where, X represents the input columns, i.e soil depth, humidity etc. and y represents the output parameter, i.e the probability of the presence of life.
3. Here, we make the differentiation of training and testing set, through the train_test_spilt function, within our training set itself. We have assigned the training set to be 80% percent of the given dataset, and 20% to be the test set, and we have made the random state as 42.
4. These lines are used as a preprocessing of the dataset, wherein we use the StandardScaler library, and it is used to remove the mean, and scale the dataset to the variance. Both the training and testing values of X are scaled. Scaling is a important tool as it helps the model to be more stable, and converge quicker. It also helps in making sure that all the imput features have the same influence on the model
5. Here we create the model, named as "knn_model" and train it, with the n_neighbours set to 5. We fit the model to the training set , and make the prediction on teh test set, and the result is stored in y_pred.
   The mean squared error is calculated, to determine the accuracy of the model

        import pandas as pd
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler
        # Assuming X1, X2, ... are your input features in the new dataset
        X_new = df1.iloc[:, :]
        
        # Standardize the features using the same scaler from the training data
        X_new_scaled = scaler.transform(X_new)  # Assuming 'scaler' is the one used during training
        
        # Make predictions on the new dataset
        y_pred_new = knn_model.predict(X_new_scaled)  # Assuming 'knn_model' is your trained KNN model
        
        # Display or use the predictions as needed
        print("Predictions for the new dataset:")
        print(y_pred_new)
   These lines are to test the model:
   1. We import all the necessary libraries, and then load the new dataframe, i.e "Lablled2.csv" and give all its input columns as features.
   2. We scale the same, and insert the data in the existing model, which gives us the predicted output, i.e y_pred and our model is complete.
------------------------------------------------------------------------------
