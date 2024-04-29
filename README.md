# potato-disease-classification
Potato disease classification using potato leaf images. 

## Setup for Python:
Install Python (Setup instructions)

Install Python packages

pip3 install -r training/requirements.txt
pip3 install -r api/requirements.txt


Training the Model
Download the data from kaggle.
Only keep folders related to Potatoes.
Run Jupyter Notebook in Browser.
jupyter notebook
Open training/potato-disease-training.ipynb in Jupyter Notebook.


Copy the model generated and save it with the version number in the saved_models folder.

## Running the API
Using FastAPI
Get inside api folder
cd api
Run the FastAPI Server using uvicorn
uvicorn main:app --reload --host 0.0.0.0
Your API is now running at 0.0.0.0:8000


