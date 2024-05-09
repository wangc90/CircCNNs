Data and code for "CircCNNs, a convolutional neural network framework to better understand the biogenesis of exonic circRNAs"

The figure below shows the rationale of how we created exon pairs that either participate in back-splicing (BS) to form exonic circRNAs or linear-splicing (LS) to form linear transcript. Our CircCNNs framework is then based on this curated dataset to classify between the BS and LS exon pairs.

![CircCNN Base models](Images/BS_LS_exon_pairs.png)


The figure below shows our base models processing the junction sequences to classify BS and LS exon pairs.
![CircCNN Base models](Images/base_models.jpg)


The figure below shows our fast numerical methods to calculate reverse complementary matches (RCMs) crossing the flanking introns or within the flanking intron.
![CircCNN Base models](Images/RCM_algorithm.png)
