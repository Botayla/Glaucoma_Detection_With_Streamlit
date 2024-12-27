Glaucoma Detection with CNN This project focuses on detecting glaucoma from fundus images using a Convolutional Neural Network (CNN). The model is designed to classify images as either positive (glaucoma detected) or negative (no glaucoma detected). The project also includes data preprocessing, visualization, and performance evaluation.
Table of Contents Overview Features Installation Usage Dataset Model Architecture Evaluation Metrics Results Contributing License Overview Glaucoma is a leading cause of blindness worldwide. Early detection is crucial for effective treatment. This project leverages deep learning techniques to build a model capable of classifying fundus images into glaucoma-positive or glaucoma-negative categories.
Features Preprocessing and augmentation of fundus images.Custom CNN architecture for feature extraction and classification.Visualization of dataset distributions and sample images.Evaluation of model performance using accuracy, precision, recall, F1-score, and confusion matrix.Deployment-ready code using Streamlit for an interactive interface.Installation Clone the repository:bashCopy codegit clone https://github.com/your-username/glaucoma-detection-cnn.git
cd glaucoma-detection-cnn
Install the required dependencies:bashCopy codepip install -r requirements.txt
Install Streamlit if not already installed:bashCopy codepip install streamlit
UsagePrepare the dataset: Place the training, validation, and test images in the respective directories as expected by the code. Update the paths in the script if necessary.Run the application:bashCopy codestreamlit run app.py
Train the model: Use the provided code to train the CNN model on the dataset.Visualize results: The Streamlit application will display sample images, class distributions, and evaluation metrics.DatasetThe project uses fundus images for glaucoma detection. Ensure the dataset is structured with labeled images.
Example directory structure:
bashCopy codedata/
├── Fundus_Train_Val_Data/
│   ├── Train/
│   ├── Validation/
│   ├── Test/
├── glaucoma.csv  # Contains image filenames and labels

Model ArchitectureThe CNN model consists of:
Convolutional layers with Batch Normalization, ReLU activation, and MaxPooling.Fully connected layers with Dropout for regularization.The final output layer for binary classification.Evaluation MetricsThe model is evaluated using the following metrics:
Accuracy Precision Recall F1-Score Confusion Matrix Results Add results once available, e.g.:
Accuracy: 0.7629, Precision: 0.5000, Recall: 0.1304, F1-Score: 0.2069 Confusion matrix and other plots are visualized using the Streamlit application.
ContributingContributions are welcome!
Fork the repository.Create a new branch: git checkout -b feature-name.Commit your changes: git commit -m 'Add feature-name'.Push to the branch: git push origin feature-name.Open a pull request.LicenseThis project is licensed under the MIT License. See the LICENSE file for details.
AcknowledgmentsInspiration: The project was inspired by the need for early glaucoma detection.Tools: Built using PyTorch, Streamlit, and supporting Python libraries.This structure provides clarity and makes your project appealing to collaborators and potential users.
