# ZenerImagesClassification

To implement the S-K Algorithm for training SVMs as well as gain familiarity with Numpy and Pillow.

Step 1
A dataset generator which can be used to generator variations of Zener Cards. These are images with various symbols on them which are then classified.

python zener_generator.py folder_name num_examples 

All 25x25 black-and-white png image files are saved in the folder folder_name. The files should have file names of the form Num_ZenerCardLetter.png. For example, 1_O.png. The number 1 means it was the first example image generated. The second image would use the number 2, and so on. The letter O means the image in the file was supposed to be a O (circle). Other single letter codes for the remaining Zener Cards are:

P - for Plus symbol
W - for Waves
Q - for Square
S - for Star


Random mutations of the same images are generated for data augmentation.

Step 2
An SVM model training program.

python sk_train.py epsilon max_updates class_letter model_file_name train_folder_name 

Step 3
An SVM model testing program. 

python svm_model_tester.py model_file_name train_folder_name test_folder_name 
Sample output
Fraction Correct: .96
Fraction False Positive: .3
Fraction False Negative: .1

Step 4
Conduct experiments



