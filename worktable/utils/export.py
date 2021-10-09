
import numpy as np
from skimage import measure, morphology

#takes in an image slice and returns a list of all contours indexed by the class labels
def get_contours(img, num_classes, level=None):
    contour_list = []
    for class_id in range(num_classes):
        out_copy = img.copy()
        out_copy[out_copy != class_id] = 0
        outed = out_copy
        contours = measure.find_contours(outed, level)
        contour_list.append(contours)
    return contour_list

#Create contour set for the entire volume: expected input in a volume of size (Z,Y,X)
#label_names are the names corresponding to each label index in model output ["background", "brain", "optic chiams", ...]
#                                                                               0            1           2       etc
#len(label_names) should equal num_classes
#Return format:
# {
#   "Brain":
#   [
#     [(x, y, z1), (x, y, z1), (x, y, z1), (x, y, z1)], # slice 1
#     [(x, y, z2), (x, y, z2), (x, y, z2), (x, y, z2)], # slice 2
#     [(x, y, z3), (x, y, z3), (x, y, z3), (x, y, z3)], # slice 3
#     .
#     .
#     .
#     [(x, y, zn), (x, y, zn), (x, y, zn), (x, y, zn)] # slice n
#   ]
# }

def mask_to_contour_set(arr, num_classes, label_names):

    #create empty dict
    contour_set = {}
    for label in label_names:
        contour_set[label] = []

    #Go though each slice
    for z_value, arr_slice in enumerate(arr):
        contours_list = get_contours(arr_slice, num_classes) #get all contours for this slice

        #Go through each label's contours
        for label, site_contours in enumerate(contours_list):
            #Go through each contour in that specific label and append the z value
            for contour_id, contour in enumerate(site_contours):
                contours_list[label][contour_id] = np.insert(contours_list[label][contour_id], 2, z_value, axis=1) #add z value into contour coordinates

        #Append into the dictionary
        for i, label in enumerate(label_names): #for each organ
            for array in contours_list[i]:
                array = array.tolist() #convert from numpy array to python list
                contour_set[label].append(array)
                #contour_set[label] = np.append(contour_set[label], array)

    #can use json.dump() to save this as a json file
    return contour_set