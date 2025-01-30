How to use:

1. Dataset is already included in the github (in data_raw folder)

To clean and process the data, run the Data_cleanup.py file

Alternatively, if you want to preprocess the images and train a model of your own, run the GNN_Training.py folder instead.

2. If you want to use the pretrained model, run the Model_Predict.py file. It will automatically open the prediction images dataset and run the model on 10 of them
note: If you want to train your own model, you can name it in the HyperParameters.py file (MODEL_NAME)

4. If you want to train your own model with new parameters, change the parameters for processing the images or for the model in HyperParameters.py
