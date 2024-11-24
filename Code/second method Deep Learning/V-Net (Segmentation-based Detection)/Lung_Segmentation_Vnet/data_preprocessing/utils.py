import os


def file_name_path(file_dir):
    """
  Get the root path, subdirectories, and all sub-files 
  from the specified directory using the directory path to analyze, 
  returning the root path, list of subdirectories, and list of all sub-files.
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs):
            print("sub_dirs:", dirs)
            return dirs

        
def files_name_path(file_dir):
    
    for root, dirs, files in os.walk(file_dir):
        if len(files):
            print("sub_files:", files)
            return files
        
def save_file2csvv2(file_dir, file_name,label):
    """
   Save the file paths to a CSV for classification, 
   using the specified preprocess data path, output CSV name, and classification label, returning the result.
    """
    out = open(file_name, 'w')
    sub_files = files_name_path(file_dir)
    out.writelines("class,filename" + "\n")
    for index in range(len(sub_files)):
        out.writelines(label+","+file_dir + "/" + sub_files[index] + "\n")
       

def save_file2csv(file_dir, file_name):
    """
    Save the file paths to a CSV for segmentation, using the specified preprocess data path and output CSV name, returning the result.
    """
    out = open(file_name, 'w')
    sub_dirs = file_name_path(file_dir)
    out.writelines("filename" + "\n")
    for index in range(len(sub_dirs)):
        out.writelines(file_dir + "/" + sub_dirs[index] + "\n")


save_file2csv("segmentation/Mask/", "train_X_mask.csv")
