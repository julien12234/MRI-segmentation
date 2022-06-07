# libraries to import 
import os 
import SimpleITK as sitk
import subprocess
import colorama
import numpy as np

"""
This script is used to segment .nii or .nii.gz files that are located in the folder "Input/". 
The predictions will be located in the output folder "Output/", with the prefix "prediction_" in front of the input files

It will use a 3d trained nnUnet model (https://github.com/MIC-DKFZ/nnUNet).
The model is trained to segment MRI of mice
nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of Helmholtz Imaging.

Please, before running the script, make sure to:
- install all the required libraries at the right version 
- have only .nii.gz files of dimensions (96, 192, 46) in the folder_input
- empty your folder_output 

If the script does not work, you can contact me at the following e-mail address: julien.adda@epfl.ch
with a screen of your:
- folder_input 
- folder_output 
- list (ls) of your current directory 
- the error message in the terminal 

Your directory should be structured in the following way:
- CIBM_prediction.py
- input/                    !!!Input files need to be .nii.gz of dimenion (96, 192, 46)!!!
    - image1.nii.gz
    - image2.nii.gz
    - image3.nii.gz
- output/                   !!!Keep this folder empty before running the script!!!
- label/                    Contains the label of the MRI in input -> always add label_ in front of the name
    - label_image1.nii.gz
    - label_image2.nii.gz
- nnUNet/                   !!!Do not modify this folder!!!

The output folder, after prediction, will be:
- output/
    - prediction_image1.nii.gz
    - prediction_image2.nii.gz
    - prediction_image3.nii.gz

The script CIBM_prediction.py can be runned via the terminal with the following command:
"python CIBM_prediction.py"
"""

def volume_3d(image, s1, s2, s3):
    """
    Arguments:
        image - 3D image
        s1, s2, s3 - spacing by x, y and z axis --> s1*s2*s3 is voxel volume
        units - calculating number of voxels that are considered as implant
    """
    units = np.count_nonzero(image)
    return units * s1 * s2 * s3


check = input("Check label folder (Yes/No): ")
while (check != "Yes" and check != "No") :
    check = input("Please write Yes or No! Check label folder (Yes/No): ")

if check == "Yes":
    check = True 
else :
    check = False

print("********************************************************************************")
print("********************************************************************************")
print("Checking existence of folders and dimensions of data...")

# folder containing the MRI input images to be predicted 
folder_input = "input/"
# folder that will contain the prediction of the model for the files located in folder_input
folder_output = "output/"
# folder that will contain the label/ground truth of the MRI in folder input 
folder_name_label = "label/"
# name of the folder containing the model -> do not touch!!
folder_model = "nnUNet/"

# check if folder_input is a folder and exists
if not os.path.isdir(folder_input):
    raise ValueError("Folder of .nii files does not exist: {}".format(folder_input))

# check if folder_model is a folder and exists
if not os.path.isdir(folder_model):
    raise ValueError("Folder of the model nnUNet does not exist: {}".format(folder_input))

# check if folder_output is a folder and exists
if not os.path.isdir(folder_output):
    raise ValueError("Folder for saving the predictions does not exist: {}".format(folder_output))

if check:
    if not os.path.isdir(folder_name_label):
        raise ValueError("Folder of labels file does not exist: {}".format(folder_name_label))

# check if folder_input is not empty 
if len(os.listdir(folder_input)) == 0:
    raise ValueError("Folder of .nii files is empty: {}".format(folder_input))

files_to_predict = []
files_to_check = []
# iterate through all the files in folder_input
for filename in os.listdir(folder_input):
    if filename == ".DS_Store":
        continue
    # check if file is a .nii file or .nii.gz file
    elif (filename[-4:] == ".nii"):
        raise ValueError("File .nii {} needs to be a .nii.gz file in folder Input!".format(filename))
    elif (filename[-7:] == ".nii.gz"):
        n2_img = sitk.ReadImage(folder_input + filename)
        # check if dimension of file is the one expected by the model 
        if (96, 192, 46) != n2_img.GetSize():
            print("Warning! file {} of dimension {} has not the expected dimensions (96, 192, 46)".format(filename, n2_img.GetSize(),))
        files_to_predict.append(filename)
        if check:
            if os.path.exists(folder_name_label+'label_'+ filename.split(".nii")[0] + ".nii"): 
                files_to_check.append(filename.split(".nii")[0] + ".nii")
            if os.path.exists(folder_name_label+'label_'+ filename.split(".nii")[0] + ".nii.gz"):
                files_to_check.append(filename.split(".nii")[0] + ".nii.gz")
    else:
        raise ValueError("File {} needs to be a .nii.gz file in folder Input!".format(filename))

# check if folder_input contains at least one .nii file or .nii.gz file 
if len(files_to_predict) == 0:
    raise ValueError("Out of the {} files to predict in folder {}, none where .nii or .nii.gz files!".format(len(os.listdir(folder_input)),folder_input))

if (len(files_to_check) == 0 and check == True) :
    raise ValueError("No labeled files where found for the {} files to predict! Check the folder Label/, and make sure that the labels are named label_X with X the name of the file in folder Input/.".format(len(files_to_predict)))

print("All good!")

print("\n")
print("Ready to predict {} files located in {}".format(len(files_to_predict),folder_input))
print("Names of files are: {}".format(files_to_predict))

print("\n")
if check :
    print("Found {} labeled files that match the input files:  {}".format(len(files_to_check),files_to_check))
print("Predictions will be located in folder: {}".format(folder_output))

print("********************************************************************************")
print("********************************************************************************")

print("Running model 3d nnUNet...")

# set variables for model nnUNet
os.environ["nnUNet_raw_data_base"] = "nnUNet_raw_data_base"
os.environ["nnUNet_preprocessed"] = "nnUNet_preprocessed"
os.environ["RESULTS_FOLDER"] = "RESULTS_FOLDER"

# task of the model (Task503_CIBM) -> !! do not change !!
task = "Task503_CIBM"

# model (3d_fullres) -> !! do not change !!
model = "3d_fullres"

for filename in files_to_predict:
    os.system("mv " + folder_input+"/"+filename +" "+ folder_input+"/"+filename.split(".nii")[0]+"_0000.nii.gz")

# in path nnUNet/ run the following command -> predict the files in -i and output the result in -o 
#p = subprocess.Popen(["nnUNet_predict", "-i../Input", "-o../Output","-t"+ task,"-f 2","-m"+ model], cwd="nnUNet/")
p = subprocess.Popen(["nnUNet_predict", "-i../input", "-o../output","-t"+ task,"-chk","model_best","-m"+ model], cwd="nnUNet/")

print(p)
p.wait()

print("\n")
print("********************************************************************************")
print("********************************************************************************")

for filename in files_to_predict:
    os.system("mv " + folder_input+"/"+filename.split(".nii")[0]+"_0000.nii.gz" +" "+ folder_input+"/"+filename)

# count the number of predicted file are contained in the folder_output
count_prediction = 0 

# iterate through all the files in folder_output
for filename in os.scandir(folder_output):
    if filename.is_file():
        # delete file named plans.pkl
        if (filename.name == "plans.pkl"):
            os.system("rm " +folder_output+"/"+filename.name)
        # delete file named postprocessing.json
        if (filename.name == "postprocessing.json"):
            os.system("rm " +folder_output+"/"+filename.name)
        # do nothing if file is named .DS_Store
        elif (filename.name == ".DS_Store"):
            continue
        elif ((filename.name[-3:] == ".gz") & (len(filename.name)<11)):
            # rename by adding prefix: "prediciton_"
            os.system("mv " + folder_output+"/"+filename.name +" "+ folder_output+"/prediction_"+filename.name)
            count_prediction = count_prediction+1
        elif ((filename.name[-3:] == ".gz") & (filename.name[:11] != "prediction_")):
            # rename by adding prefix: "prediciton_"
            os.system("mv " + folder_output+"/"+filename.name +" "+ folder_output+"/prediction_"+filename.name)
            count_prediction = count_prediction+1
        else :
            continue 

print("\n")
if (count_prediction == 0):
    print(colorama.Fore.RED + "No files were predicted out of the {} files. End of code: Failure".format(len(files_to_predict)) + colorama.Fore.RESET)
elif (count_prediction == len(files_to_predict)):
    print(colorama.Fore.GREEN + "Prediction done. End of code: Success." + colorama.Fore.RESET +" All {} files were predicted.".format(len(files_to_predict)))
    print("Prediction are located in folder: {}".format(folder_output))

    print("\n")
    if check:
        print("Checking result of the model ...")
        prediction_mean = []

        for file in files_to_check:
            path_to_mask_nii = "Label/label_" + file
            path_to_prediciton_nii = "Output/prediction_" + file.split(".nii")[0] + ".nii.gz"

            scan_mask = sitk.ReadImage(path_to_mask_nii)
            space_mask = scan_mask.GetSpacing()
            spacing_mask_x = space_mask[0]
            spacing_mask_y = space_mask[1]
            spacing_mask_z = space_mask[2]
            scan_mask = sitk.GetArrayFromImage(scan_mask)

            scan_pred = sitk.ReadImage(path_to_prediciton_nii)
            space_pred = scan_pred.GetSpacing()
            spacing_pred_x = space_pred[0]
            spacing_pred_y = space_pred[1]
            spacing_pred_z = space_pred[2]
            scan_pred = sitk.GetArrayFromImage(scan_pred)

            if scan_mask.shape != scan_pred.shape:
                    raise Exception("The dimensions {} of the label file {} are different from the dimensions of the predicted file of the model {}".format(scan_mask.shape,file,scan_pred.shape))
            
            if space_pred != space_mask:
                raise ValueError("Spacing are not the same between prediction output file {} and label file {}! {} vs {}!".format(file.split("label_")[1], file, space_pred, space_mask))

            real_volume = volume_3d(scan_mask, spacing_mask_x, spacing_mask_y, spacing_mask_z)
            predicted_volume = volume_3d(scan_pred, spacing_pred_x, spacing_pred_y, spacing_pred_z)

            if real_volume == 0:
                raise ValueError("Volume of labeled file {} in Label/ is 0!".format(file))

            prediction_mean.append(100*(predicted_volume-real_volume)/real_volume)

            print("File {} -> real_volume: {:.2f}    predicted_volume: {:.2f}    diff: {:.2f}    diff_%: {:.2f}".format(file,real_volume, predicted_volume, predicted_volume-real_volume, 100*(predicted_volume-real_volume)/real_volume))
    
    print("\n")
    print("For {} files: absolute_mean_error_%: {:.2f}    std_error_%: {:.2f}    max_error_%: {:.2f}    mean_error_%: {:.2f}".format(len(files_to_check),np.mean(np.abs(prediction_mean)),np.std(np.abs(prediction_mean)),np.max(np.abs(prediction_mean)),np.mean(prediction_mean)))

elif (count_prediction < len(files_to_predict)):
    print(colorama.Fore.GREEN + "Prediction done. End of code: Success." + colorama.Fore.RESET +" WARNING: Only {} files out of {} were predicted. Unabled to compare with label folder.".format(count_prediction,len(files_to_predict)))
    print("Prediction are located in folder: {}".format(folder_output))
else :
    print("Something went wrong. Check input files and empty output folder")
print("\n")


       