import numpy as np
import math
import pandas as pd

import cv2, time

def to_tensor(a):
    return tf.convert_to_tensor(a, dtype=tf.float32)

def find_coordinates_numpy(points):
    # Take in tensor of points (flattened)
    # Return list of 2D coords
    if tf.equal(tf.size(points), 0):
        return 0
    coord = []
    length = int(math.sqrt(points.size)) # to the square root of the second dimension of shape
    print("length",length)
    for idx in points:
        i = idx//length
        j = idx % length 
        coord.append([i,j])
    return np.asarray(coord)

def find_mean_distance(coord_A, coord_B):
    n_coord = len(coord_A)
    if n_coord < 10:
        return 'too few'
    for point in coord_A:
        distances = []
        min_dist = 1000
        for coord in coord_B:
            diff = np.subtract(coord,point)
            dist = math.sqrt(diff[0]**2 + diff[1]**2)
            if dist < min_dist:
                min_dist = dist
        distances.append(min_dist)
    distances = np.asarray(distances)
    return np.mean(distances)

def find_coordinates_numpy(points):
    # Take in tensor of points (flattened)
    # Return list of 2D coords
    if points.sum() < 20:
        return 'too few'
    coord = []
    length = 60 # to the square root of the second dimension of shape
    for idx, point in enumerate(points):
        if point > 0:
            i = idx//length
            j = idx % length
            coord.append([i,j])
    return np.asarray(coord)

def MSE_plus_MED(cutoff=25.4,constant_loss=1):
    # medMiss gives mean distance of misses in A, by B
    def loss_function(y_true, y_pred):
        MSE = tf.math.reduce_mean(tf.square(y_true - y_pred))
        target_tensor = tf.reshape(y_true, (tf.shape(y_true)[0], -1)).numpy() # reshape for sample-wise
        prediction_tensor = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1)).numpy()
        # hard discretization
        target_tensor = tf.cast(tf.where(target_tensor<cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(prediction_tensor<cutoff,0.0,1.0),tf.float32)
        # Now calculate MED
        MED = 0 # initialize MED
        for true, pred in zip(target_tensor, prediction_tensor):
            points_A = tf.where(true).numpy() # get indices of nonzero elements and convert to numpy
            points_B = tf.where(pred).numpy()
            coord_A = find_coordinates_numpy(points_A) # get coordinates of nonzero elements on 2D grid
            coord_B = find_coordinates_numpy(points_B)
            if not isinstance(coord_A, str) and not isinstance(coord_B, str): # There are points in both A and B
                MED = MED + find_mean_distance(coord_A,coord_B)
            if not isinstance(coord_A, str) and isinstance(B, str): # There are points in A but not in B
                MED = MED + constant_loss # for now, just add a constant if B == 0
            else: # A is completely below threshold
                MED = MED + 0 # if A is below threshold, just use MSE (i.e. MED = 0)
        return MSE + MED
    return loss_function

def MSE_plus_medMiss(cutoff=25.4, constant_loss=1):
    def medMiss(y_true, y_pred):
        MSE = tf.math.reduce_mean(tf.square(y_true - y_pred))
        print(MSE)
        target_tensor = tf.reshape(y_true, (tf.shape(y_true)[0], -1)).numpy() # reshape for sample-wise
        prediction_tensor = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1)).numpy()
       # MSE = K.sqrt(tf.math.reduce_mean(tf.square(target_tensor - prediction_tensor), axis = [1])) # calculate sample-wise MSE
        # hard discretization
        target_tensor = tf.cast(tf.where(target_tensor<cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(prediction_tensor<cutoff,0.0,1.0),tf.float32)
        # Now calculate MED
        MED = 0 # initialize MED
        for idx, sample in enumerate(target_tensor):
            points_A = tf.where(sample).numpy() # get indices of nonzero elements and convert to numpy
            points_B = tf.where(prediction_tensor[idx]).numpy()
            coord_A = find_coordinates_numpy(points_A) # get coordinates of nonzero elements on 2D grid
            coord_B = find_coordinates_numpy(points_B)
            if coord_A != 0 and coord_B != 0:
                MED = MED + find_mean_distance(coord_B,coord_A)
            if coord_A != 0 and coord_B == 0: # check for all FNs in B
                # mupltiply by constant loss * number of FNs
                MED = MED + constant_loss # for now, just add a constant if B == 0
            else:
                MED = MED + 0 # if A is below threshold, just use MSE (i.e. MED = 0)
            print(MED)
        return MSE + MED
    return medMiss

def myMED(cutoff=20):
    def mean_error_distance(y_true, y_pred):
        # Sample specific MED (i.e. not whole batch at once)
        # Either maxpool or limit domain size or threshold by high val to reduce computation
        target_tensor = tf.cast(tf.where(y_true<cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(y_pred<cutoff,0.0,1.0),tf.float32)
        target_tensor = tf.reshape(target_tensor, (tf.shape(target_tensor)[0], -1)).numpy()
        prediction_tensor = tf.reshape(prediction_tensor, (tf.shape(prediction_tensor)[0], -1)).numpy()
        MED = 0
        for idx, sample in enumerate(target_tensor):
            points_A = tf.where(sample).numpy() # get indices of nonzero elements and convert to numpy
            points_B = tf.where(prediction_tensor[idx]).numpy()
            coord_A = find_coordinates_numpy(points_A) # get coordinates of nonzero elements on 2D grid
            coord_B = find_coordinates_numpy(points_B)
            if coord_A != 0 and coord_B != 0:
                print('Coordinates are present, calculating MED')
                MED = MED + find_mean_distance(coord_A,coord_B)
                print('MED {}'.format(MED))
            elif coord_A == 0 and coord_B != 0:
                MED = MED + K.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred), axis = [1]))
                print('A completely below threshold with {}'.format(MED))
            elif coord_A != 0 and coord_B == 0:
                MED = MED + K.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred), axis = [1]))
                print('B completely below threshold with {}'.format(MED))
            else:
                MED = MED + K.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred), axis = [1]))
                print('Triggered else clause with {}'.format(MED))
            #except IndexError:
            #    print('index errror')
            #    MED = MED + 0
        return MED
    return mean_error_distance

def distance(point1, point2):
    # get x and y as return from point tuple
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def get_distance_slow(A, B):
    coord_A = get_coord(A)
    coord_B = get_coord(B)
    min_dists = []
    for idx in range(len(coord_A)):
        distances = []
        for idx2 in range(len(coord_B)):
            distance = distance(coord_A[idx], coord_B[idx2])
            distances.append(distance)
        min_dists.append(min(distances))
    return min_dists



def get_coord(A,method='numpy'):
    if method == 'numpy':
        indices = np.where(A)
        points = [(x,y) for x,y in zip(indices[0],indices[1])]
        return points

def is_in_elipse(point, elipse_center, elipse_radius):
    x, y = point
    cx, cy = elipse_center
    rx, ry = elipse_radius
    return (x-cx)**2/(rx**2) + (y-cy)**2/(ry**2) <= 1

def make_elipse(center, radius, n=60):
    #  A = m.make_elipse((15,20), (10,8))
    #  B = m.make_elipse((13,25), (11,15))
    A = np.ones((n,n))
    indices = np.where(A)
    points = [(x,y) for x,y in zip(indices[0], indices[1])]
    in_elipse = [is_in_elipse(point, center, radius) for point in points]
    A = np.asarray(in_elipse).astype(int)
    return np.reshape(A, (n,n))

def distance_1D(x1, x2):
    # get x and y as return from point tuple
    return (x1-x2)**2 

def distance_transform(B):
    # requires B to be binary
    G = []
    for row in B:
        # convert row to list
        # get indices of 0s in row
        zeros = np.where(row == 0)[0]
        ones = np.where(row == 1)[0]
        G_row = []
        for idx, value in enumerate(row):
            if idx in ones:
                if len(zeros) > 0:
                    dist = min(distance_1D(idx, zeros))
                else:
                    dist = 60**2

            else:
                dist = 0
            G_row.append(dist)
        G.append(G_row)
    # Now G is built, as a list of rows
    # Convert to numpy array
    G = np.asarray(G)
    G = np.transpose(G)
    indices = np.arange(0, 60)
    H = np.zeros((60,60))
    for i, col in enumerate(G):
        for idx, value in enumerate(col):  
            if G[i,idx] == 0:
                continue # leave as 0 if point is above threshold in B
            col_transform = col + [distance_1D(x, idx) for x in indices]
            dist_idx = min(col_transform)
            # square root 
            H[i, idx] = np.sqrt(dist_idx)
    return H
# implement distance transofrm using bucket sorting

def min_dists(A, B, threshold=20):
    A = np.where(A<threshold, 0, 1)
    B = np.where(B<threshold, 0, 1)
    H = distance_transform(B)
    min_dists = A*H
    return min_dists  

def hausdorf(A,B):
    min_dists_AB = min_dists(A,B)
    min_dists_BA = min_dists(B,A)
    return np.max(np.max(min_dists_AB), np.max(min_dists_BA))


def samplewise_RMSE(y_true, y_pred):
    # from CIRA guide
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_true)[0], -1])
    return K.sqrt( tf.math.reduce_mean(tf.square(y_true - y_pred), axis=[1]))        

def neighborhood_stats(y_true, y_pred, thres, n):
    # Given a pixel, determine if a pixel within a 
    # n x n neighborhood is above threshold for hit,
    # if so, increment TP by 1 
    # n = 1 yields a .5 x .5 neighborhoood
    # for reference, 5 km x 5 km: village (n = 10)
    # for reference, 10 km x 10 km: city (n = 20)
    # for reference, 15 km x 15 km: county (n = 30)
    # dilate prediction by n x n filter
    for idx, image in enumerate(y_pred):
        # make image binary
        image = image[image>thres]
        truth = y_true[idx][y_true[idx]>thres]
        # dilate image by n x n filter
        dilated_image = cv2.dilate(image, np.ones((n,n), np.uint8))
        # get true positives
        true_positives = np.sum(np.logical_and(dilated_image, truth))
        # get false positives
        false_positives = np.sum(np.logical_and(dilated_image, np.logical_not(truth)))
        # get false negatives
        false_negatives = np.sum(np.logical_and(np.logical_not(dilated_image), truth))
        # get true negatives
        true_negatives = np.sum(np.logical_and(np.logical_not(dilated_image), np.logical_not(truth)))
        # calculate precision  
        precision = true_positives / (true_positives + false_positives)
        # get POD
        POD = true_positives / (true_positives + false_negatives)
        # get FAR
        FAR = false_positives / (false_positives + true_negatives)
        # get CSI
        CSI = (true_positives + false_negatives) / (true_positives + false_positives + false_negatives + true_negatives)
    return POD, FAR, CSI, precision

def CSI(cutoff=20):
    # From CIRA guide to loss functions, with slight differences        
    # Does not train well off thebat, but could be used as second phase for MSE model
    def loss(y_true, y_pred):
        target_tensor = tf.cast(tf.where(y_true>cutoff,0.0,1.0),tf.float32)
        prediction_tensor = tf.cast(tf.where(y_pred>cutoff,0.0,1.0),tf.float32)

        num_true_positives = K.sum(target_tensor * prediction_tensor)
        num_false_positives = K.sum((1 - target_tensor) * prediction_tensor)
        num_false_negatives = K.sum(target_tensor * (1 - prediction_tensor))
            
        denominator = (
            num_true_positives + num_false_positives + num_false_negatives +
            K.epsilon()
            )
        csi_value = num_true_positives / denominator
        return csi_value
        #return 1. - csi_value # subtract from 1 so we minimize i.e. increase CSI
    return loss

def FAR(cutoff=15):
    def far(y_true, y_pred):
        TP = sum(np.where(((y_true-1.0)+y_pred)<1.0,0.0,1.0))
        FP = sum(np.where(y_pred-y_true<1.0,0.0,1.0))
        return FP/(TP+FP+0.000001)
    return far

def POD(cutoff=15):
    # computes the probability of detection
    # POD = PT / TP + FN
    def pod(y_true, y_pred):
        TP = sum(np.where(((y_true-1)+y_pred)<1.0,0.0,1.0))
        FN = sum(np.where(y_true-y_pred<1.0,0.0,1.0))
        return TP/(TP+FN+0.000001)
    return pod

def hausdorf(mindists_AB,mindists_BA):
    return np.max(np.max(mindists_AB), np.max(mindists_BA))

def PHDK(mindists_AB, mindists_BA, k_pct):
    len_mindists_AB = len(mindists_AB)
    len_mindists_BA = len(mindists_BA)
    min_N = min(len_mindists_AB, len_mindists_BA)
    k = int(min_N * k_pct)
    # sort mindists_AB and mindists_BA in descending order
    mindists_AB.sort()
    mindists_BA.sort()
    kth_AB = mindists_AB[k]
    kth_BA = mindists_BA[k]
    return max(kth_AB, kth_BA)

def Gbeta(A,B, mindists_AB, mindists_BA, beta):
    n_A = np.sum(A)
    n_B = np.sum(B)
    n_AB = np.sum(A*B)
    y1 = n_A + n_B + n_AB
    med_AB = np.mean(mindists_AB) # try median
    med_BA = np.mean(mindists_BA)
    y2 = med_AB*n_B + med_BA*n_A
    y = y1*y2
    const = 1 - (y/beta)
    G_beta = max(const, 0)
    return G_beta

# find the shortest distance between two set of points as numpy arrays
def find_coordinates(A):
    # returns a tensor of coordinates of nonzero elements in A
    coord_A = []
    length = A.shape[0]
    for idx, row in enumerate(A):
        for idx2, pixel in enumerate(row):
            if pixel == 1: 
                i = idx%length
                j = idx2
                coord_A.append([i,j])  
    return coord_A

def find_shortest_distance(A,B):
    # returns a tensor of coordinates of nonzero elements in A
    coord_A = find_coordinates(A)
    coord_B = find_coordinates(B)

    point = coord_A[0]
    distances = []
    for coord in coord_B:
        distances.append(np.linalg.norm(np.subtract(coord,point)))
    return np.min(distances) 


def get_contour_set_2(A):
    # defines contour pixels as white pixels with at least one black neighbor
    # A is a 2D binary image
    x_ind, y_ind = np.where(A==1)
    white_points = [(x,y) for x,y in zip(x_ind,y_ind)]
    contour_set = []
    buckets = [[] for x in np.arange(60)]
    for x in x_ind:
        for y in y_ind:
            neighbors = get_neighbors(x,y)
            # Add neighbors to contour set if they are not in x_ind, y_ind and are not in the contour set
            for neighbor in neighbors:
                if neighbor not in white_points and neighbor not in contour_set and y < 60 and y >= 0 and x < 60 and x >= 0:
                    buckets[0].append((x,y,0,0))
                    if neighbor not in buckets[0]:
                        if neighbor[0] != x and neighbor[1] != y:
                            dx = 1
                            dy = 1
                            buckets[2].append((neighbor[0],neighbor[1],dx,dy))
                        if neighbor[0] != x:
                            dx = 1
                            dy = 0
                            buckets[1].append((neighbor[0],neighbor[1],dx,dy))
                        if neighbor[1] != y:
                            dx = 0
                            dy = 1
                            buckets[1].append((neighbor[0],neighbor[1],dx,dy))
    return buckets

def get_neighbors(x,y):
    return [[x-1,y-1],[x-1,y],[x-1,y+1],[x,y-1],[x,y+1],[x+1,y-1],[x+1,y],[x+1,y+1]]

def main():
    A = make_elipse((15,20), (10,8))
    B = make_elipse((13,25), (11,15))
    H_A = distance_transform(A)
    print(H_A)

if __name__ == '__main__':
    main()