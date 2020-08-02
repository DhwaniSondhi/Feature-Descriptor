import cv2
import numpy
import math
import sys

def adaptive_sus(points,resize):
    sorted_points=list()
    for point_1 in points:
        small_dist=sys.float_info.max
        for point_2 in points:
            distance=(((point_1.pt[0]-point_2.pt[0])**2)+((point_1.pt[1]-point_2.pt[1])**2))**0.5
            if(distance!=0) and point_1.response<(0.9*point_2.response) and distance<small_dist:
                small_dist=distance
        point_1.size=small_dist
        sorted_points.append(point_1)
    sorted_points=sorted(sorted_points, key=lambda point: point.size, reverse=True)
    if len(sorted_points)>resize:
        sorted_points=sorted_points[:resize]
    return sorted_points
	
def rotateinvariance(mag,angle):
    histogram=numpy.zeros(36)
    ret_angles=list()
    for row in range(0,angle.shape[0]):
        for col in range(0,angle.shape[1]):
            angle[row][col]%=360
            key=int(math.floor(angle[row][col]/10))
            histogram[key]+=mag[row][col]
    maxval=max(histogram)
    dst=list()
    for loop in range(0,len(histogram)):
        if histogram[loop]>=(maxval*0.8):
            dst.append(loop)
    for loc in dst:
        new_angle=angle
        for row in range(0,new_angle.shape[0]):
            for col in range(0,new_angle.shape[1]):
                new_angle[row][col]-=(loc*10)
                angle[row][col]%=360
        ret_angles.append(new_angle)
    return ret_angles
	
def create_matching(points_1, points_2):
    final_features1=list()
    final_features2=list()
    distances=list()
    for img1 in points_1:
        feat_desp1=points_1[img1]
        totals_pt=dict()
        totals=list()
        for img2 in points_2:
            total=0
            feat_desp2=points_2[img2]
            ft_desp=(feat_desp1-feat_desp2)**2
            for loop in ft_desp:
                total+=loop
            
            totals.append(total)
            totals_pt[img2]=total
        totals=sorted(totals)
        
        if totals[0]<0.5 and (totals[0]/totals[1])<0.6: 
            print(totals[0])
            final_features1.append(cv2.KeyPoint(int(img1.split()[0]),int(img1.split()[1]),1))
            ans=""
            for key in totals_pt:
                if totals_pt[key]==totals[0]:
                    ans=key
            final_features2.append(cv2.KeyPoint(int(ans.split()[0]),int(ans.split()[1]),1))
            distances.append(totals[0])
    
    loop1=0
    removeEle=set()
    for key1 in final_features1:
        loop2=0
        key1=key1.pt
        for key2 in final_features1:
            key2=key2.pt
            if loop1!=loop2 and key1[0]==key2[0] and key1[1]==key2[1]:
                if distances[loop1]>distances[loop2]:
                    removeEle.add(loop1)
                else:
                    removeEle.add(loop2)
            loop2+=1
        loop1+=1
    
    loop1=0
    for key1 in final_features2:
        loop2=0
        key1=key1.pt
        for key2 in final_features2:
            key2=key2.pt
            if loop1!=loop2 and key1[0]==key2[0] and key1[1]==key2[1]:
                if distances[loop1]>distances[loop2]:
                    removeEle.add(loop1)
                else:
                    removeEle.add(loop2)
            loop2+=1
        loop1+=1
    
    res_features1=list()
    res_features2=list()
    matchings=list()
    out_loop=0
    for loop in range(0,len(distances)):
        if loop not in removeEle:
            res_features1.append(final_features1[loop])
            res_features2.append(final_features2[loop])
            matchings.append(cv2.DMatch(out_loop,out_loop,distances[loop]))
            out_loop+=1
    
    return res_features1, res_features2, matchings

def create_mag_angle(inp_img):
    rows, cols=inp_img.shape
    inp_img = numpy.float32(inp_img)
    g_x=numpy.zeros(inp_img.shape)
    g_y=numpy.zeros(inp_img.shape)
    for row in range(1,rows-1):
        for col in range(1,cols-1):
            g_x[row][col]=inp_img[row][col+1]-inp_img[row][col-1]
            g_y[row][col]=inp_img[row+1][col]-inp_img[row-1][col]
            
    magnitude=((g_x**2)+(g_y**2))**0.5
    degrees=numpy.degrees(numpy.arctan2(g_y,g_x))
    return magnitude, degrees

def sift(input_img, points):
    features=dict()
    grey_inp = cv2.GaussianBlur(cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY),(0,0),1.5)
    magnitude,degrees=create_mag_angle(grey_inp)
    key_loop=0
    for point in points:
        pt_c=int(point.pt[0])
        pt_r=int(point.pt[1])
        if pt_r-8>=0 and pt_c-8>=0 and pt_r+8<=magnitude.shape[0] and pt_c+8<=magnitude.shape[1]:
            mag16=magnitude[pt_r-8:pt_r+8,pt_c-8:pt_c+8]
            mag16=cv2.normalize(mag16,None,norm_type=cv2.NORM_L2)
            angleout=degrees[pt_r-8:pt_r+8,pt_c-8:pt_c+8]
            array=[0,4,8,12]
            ret_angles=rotateinvariance(mag16,angleout)
            ##multiple keypoints
            for angle16 in ret_angles:
                descrip_128=list()
                for r in array:
                    for c in array:
                        histogram=numpy.zeros(8)
                        window_mag=mag16[r:r+4,c:c+4]
                        window_angle=angle16[r:r+4,c:c+4]
                        for i in range(0,4):
                            for j in range(0,4):
                                window_angle[i][j]%=360
                                key=int(math.floor(window_angle[i][j]/45))
                                histogram[key]+=window_mag[i][j]

                        descrip_128.extend(list(histogram))
                ##normalise descriptor
                descrip_128 = numpy.clip(descrip_128, a_min=0,a_max=0.2)
                descrip_128= numpy.array(descrip_128)
                descrip_128 = cv2.normalize(descrip_128, None, norm_type=cv2.NORM_L2)
                keyName=str(pt_c)+" "+str(pt_r)+" "+str(++key_loop)
                features[keyName]=descrip_128
       
    return features
	
def create_harris_matrix(input_img,k_value,threshold):
    input_gray=input_img
    derivative_x=cv2.Sobel(input_gray,cv2.CV_32F,1,0,ksize=5)
    derivative_y=cv2.Sobel(input_gray,cv2.CV_32F,0,1,ksize=5)
    I_x2=derivative_x**2
    I_y2=derivative_y**2
    I_xy=derivative_x*derivative_y
    G_x2=cv2.GaussianBlur(I_x2,(3,3),0)
    G_y2=cv2.GaussianBlur(I_y2,(3,3),0)
    G_xy=cv2.GaussianBlur(I_xy,(3,3),0)
    I_x2_paded=cv2.copyMakeBorder(I_x2, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)
    I_y2_paded=cv2.copyMakeBorder(I_y2, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)
    I_xy_paded=cv2.copyMakeBorder(I_xy, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)
    corner_M=numpy.zeros(I_xy.shape)
    for row in range(2,I_x2_paded.shape[0]-2):
        for col in range(2,I_x2_paded.shape[1]-2):
            inner_x2=I_x2_paded[row-2:row+3,col-2:col+3]
            inner_y2=I_y2_paded[row-2:row+3,col-2:col+3]
            inner_xy=I_xy_paded[row-2:row+3,col-2:col+3]
            
            sum_x2=numpy.sum(inner_x2)
            sum_y2=numpy.sum(inner_y2)
            sum_xy=numpy.sum(inner_xy)
            
            determinant=(sum_x2*sum_y2)-(sum_xy**2)
            trace=sum_x2+sum_y2
            
            corner=determinant/(k_value+trace)
            if corner>threshold:
                corner_M[row-2][col-2]=corner
    return corner_M
    
def max_suppression_with_scale(zero_corner,threshold):
    rows,cols=zero_corner[0].shape
    points=list()
    for row in range(0,rows-3):
        for col in range(0,cols-3):
            a_max=[None]*3
            a_max[0]=zero_corner[0][row:row+3,col:col+3]
            a_max[1]=zero_corner[1][row:row+3,col:col+3]
            a_max[2]=zero_corner[2][row:row+3,col:col+3]
            check_val=a_max[1][1][1]
            ##1 max, 0 min, -1 none
            min_max=-1
            for loop in range(0,3):
                if loop==1:
                    continue
                min_val,max_val,_,_=cv2.minMaxLoc(a_max[loop])
                if check_val>max_val and (min_max==1 or min_max==-1):
                    min_max=1
                elif check_val<min_val and (min_max==0 or min_max==-1):
                    min_max=0
                else:
                    min_max=-1
            
            if min_max==-1:
                continue
            min_val,max_val,min_pos,max_pos=cv2.minMaxLoc(a_max[1])
            bool1=(min_max==1 and max_pos[0]==1 and max_pos[1]==1 and check_val>threshold)
            bool2=(min_max==0 and min_pos[0]==1 and min_pos[1]==1 and check_val>threshold)
            if bool2 or bool2:
                keyPt=cv2.KeyPoint(col,row,1)
                keyPt.response=check_val
                points.append(keyPt)
                
    return points

def finding_points(corner_list,threshold):
    zero_corner=[None]*3
    zero_corner[0]=cv2.copyMakeBorder(corner_list[0], 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    zero_corner[1]=cv2.copyMakeBorder(corner_list[1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    zero_corner[2]=cv2.copyMakeBorder(corner_list[2], 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    
    return max_suppression_with_scale(zero_corner,threshold)
	
def downsampling(inp_img):
    updated_dimensions=(int(inp_img.shape[1]*0.5),int(inp_img.shape[0]*0.5))
    return cv2.resize(inp_img, updated_dimensions, interpolation = cv2.INTER_AREA)
	
def creating_octave(inp_img,siGma,s):
    dogArray=list()
    k=2**(1.0/s)
    for loop in range(0,s+1):
        G1=cv2.GaussianBlur(inp_img,(0,0),siGma)
        siGma=siGma*k
        G2=cv2.GaussianBlur(inp_img,(0,0),siGma)
        dogArray.append(G2-G1)
    return dogArray

def scale_invariance(inp_img,k_value,threshold,adapt_resize):
    grey_inp=cv2.cvtColor(inp_img,cv2.COLOR_BGR2GRAY)
    siGma=1.5
    s=3
    list_oct_harris=list()
    list_octaves=list()
    points=list()
    inp_octave=grey_inp
    for oct_loop in range(0, 3):
        octave=creating_octave(inp_octave, siGma, s)
        list_octaves.append(octave)
        harris_list=list()
        for dog in octave:
            print("Octave")
            harris_list.append(create_harris_matrix(dog,k_value,threshold))
        list_oct_harris.append(harris_list)
        inp_octave=downsampling(inp_octave)
    loop=0
    for oct_loop in list_octaves:
        for dog in oct_loop:
            name="Loop"+str(loop)+".png"
            cv2.imwrite(str(name),dog)
            loop+=1
            
    for harris_loop in list_oct_harris:
        for loop in range(1,len(harris_loop)-1):
            points.extend(finding_points(harris_loop[loop-1:loop+2],threshold))
    
    check_img=numpy.zeros(grey_inp.shape)
    for point in points:
        row=int(point.pt[1])
        col=int(point.pt[0])
        check_img[row][col]=point.response
    
    print(len(points))
    points=list()
    for row in range(0,check_img.shape[0]):
        for col in range(0,check_img.shape[1]):
            if check_img[row][col]>0:
                keyPt=cv2.KeyPoint(col,row,1)
                keyPt.response=check_img[row][col]
                points.append(keyPt)
    print(len(points))
    points=adaptive_sus(points,adapt_resize)
    return points
	
k_value=0.000000000000000001
adapt_resize=500
threshold=40000000
input_img_1=cv2.imread("pano1_0008.png")
points_1=scale_invariance(input_img_1, k_value, threshold, adapt_resize)
out_img1=cv2.drawKeypoints(input_img_1, points_1, None, color=(0,255,0), flags=0)
cv2.imwrite("abc.png",out_img1)
feat_dsp_1=sift(input_img_1,points_1)

input_img_2=cv2.imread("pano1_0009.png")
points_2=scale_invariance(input_img_2, k_value, threshold, adapt_resize)
out_img2=cv2.drawKeypoints(input_img_2, points_2, None, color=(0,255,0), flags=0)
cv2.imwrite("abc2.png",out_img2)
feat_dsp_2=sift(input_img_2,points_2)


final_features1, final_features2, matchings=create_matching(feat_dsp_1, feat_dsp_2)

out_img=cv2.drawMatches(input_img_1,final_features1,input_img_2,final_features2, matchings, None, flags=2)
cv2.imwrite("output.png",out_img)
