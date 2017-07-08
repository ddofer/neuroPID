Instructions for Running NeuroPID:

Foreword:
This implements the neuropeptide precursor protein prediction method, as seen in:
    1. Ofer, D. & Linial, M. NeuroPID: a predictor for identifying neuropeptide precursors from metazoan proteomes. Bioinformatics 30, 931–40 (2014).
    2. Karsenty, S., Rappoport, N., Ofer, D., Zair, A. & Linial, M. NeuroPID: a classifier of neuropeptide precursors. Nucleic Acids Res. (2014). doi:10.1093/nar/gku363

We request that you cite our papers if you use it! Thanks!

For "casual" users, we highly recomend using the website: http://neuropid.cs.huji.ac.il/
The website is easier to use, and implements multiple classifiers and additional predictors.



To run the program:

1. Place the (multi)-Fasta file containing all your target sequences in the "TEST_Data" directory.
2. Run (locally), using Python, the file "FeatureGen.py" within the "TEST_Data" directory.
3. In the main directory, using Python run the file "Test_OrganismML.py".
This will output a set of .csv files with predictions for each sequence, ordered in the original fasta files order.

Users using Python 2.7 may find the file "getPredictedOrganismResults.py" of use to get names of sequences matching the predictions. This (getPredictedOrganismResults.py) is not a fully tested script, so it may require tweaking to work properly.

Finally: Be careful to use sequences without any "illegal" or non standard aa (X,Z, etc' )!  Such sequences are removed during the feature generation stage, and will cause a mismatch when matching up predictions with the originating sequences.

(Also, ensure you have all the relevant packages installed. We recommend using the Anaconda 2.7 distribution)


For further details, questions and comments, feel free to contact us!
Dan Ofer
DDOFER@GMAIL.COM
