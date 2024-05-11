# "CircCNNs, a convolutional neural network framework to better understand the biogenesis of exonic circRNAs"

## Organization of this Repository

### The Data folder:
contains the genomic coordinates of all the back-splicing (BS), and linear-splicing (LS) exon pairs used in this study. Due to the file size limitation in GitHub, the remaining dataset can be downloaded from https://drive.google.com/drive/folders/1wq6vJ83Bjy9g0zlWZklpU-hEx1xHWhDG.

### The Model_Codes folder:
contains the model specifications and training process for the base, RCM as well as Combined model structures. It also contains the retraining code to retrain the base, RCM as well as Combined models. 

### The Auxiliary_Codes folder:
contains data preparation code, model training cross-validation code, position probability matrix extraction code, fast numerical method for RCM calculation code as well as a Jupyter notebook to extract the motif from the trained optimal base models.

### The Model_Evaluation folder contains:
a Jupyter notebook for model evaluation on the testing set. 

### The Trained_Model_Weights folder contains:
The trained model weights that can be used to evaluate the model performance on the testing set by using the Jupyter notebook in the Model_Evaluation folder.


1. The figure below shows the rationale of how we created exon pairs that either participate in back-splicing (BS) to form exonic circRNAs or linear-splicing (LS) to form linear transcript. Our CircCNNs framework is then based on this curated dataset to classify between the BS and LS exon pairs.

![CircCNN Base models](Images/BS_LS_exon_pairs.png)


2. The figure below shows our base models processing the junction sequences to classify BS and LS exon pairs.
   
![CircCNN Base models](Images/base_models.jpg)


3. The figure below shows our fast numerical methods to calculate reverse complementary matches (RCMs) crossing the flanking introns or within the flanking intron.
   
![CircCNN Base models](Images/RCM_algorithm.png)

### Making predictions for exon pairs being BS or LS
To make a prediction for the given exon pairs of this format:
chr19|58921331|58929694|+<br>
chr9|91656940|91660748|-<br>
chr19|5724818|5768253|+<br>

Follow the example in the testing.py within the Inference folder, and change the path for your testing dataframe, 
you can also use other trained model weights in this study to do the prediction.

The results will be similar like this:
![image](https://github.com/wangc90/CircCNNs/assets/54656523/dfce1f60-c8c6-4022-af53-2082ee48e6d9)




#### If you find anything useful here, please cite our work.
