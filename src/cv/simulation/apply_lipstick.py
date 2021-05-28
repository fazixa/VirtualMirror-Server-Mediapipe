from __future__ import division
import cv2
from matplotlib.colors import LightSource
import numpy as np
from numpy.linalg import eig, inv
from numpy.ma.core import maximum_fill_value
from scipy.interpolate import interp1d, splprep, splev
from scipy.interpolate import InterpolatedUnivariateSpline
from pylab import *
from skimage import color
from scipy import misc
import scipy.misc
import time
import imageio

class Lipstick(object):

    def __init__(self):
        """ Initiator method for class """
        self.red_l = 0
        self.green_l = 0
        self.blue_l = 0

        self.image = 0
        self.im_copy = 0
        


        self.intensity = 0
        self.x = []
        self.y = []
        self.intensitymoist = 0.8
        self.x_all = []
        self.y_all = []

    def get_lips(self,x, y):

        """ Seprates corresponding lips points from all detected points

        Args:
            param1 : X cordinates of the face.
            param2 : Y cordinates of the face.

        Returns: 
            a 2D array containing X and Y cordinates of the detected points of the lips

        note:
            lips points detected by DLib consists of 24 points with the first 12 points indicating outter parts of the lips and the rest indicating inner parts.
        """
        points = []
        for i in range(0, len(x) - 1):
            if i >= 48 and i <= 59:
                pos = (x[i], y[i])
                points.append(pos)
        pos = (x[48], y[48])
        points.append(pos)
        for i in range(0, len(x) - 1):
            if i >= 60 and i <= 67:
                pos = (x[i], y[i])
                points.append(pos)
                if i == 64:
                    pos = (x[54], y[54])
                    points.append(pos)
        pos = (x[67], y[67])
        points.append(pos)
        return points
    

    def draw_curves(self, points):

        """  Draws 4 curves given the detected points of the lips by creating an interpolated path.

        Args:
            param1 :  a 2D array containing X and Y cordinates of the detected lips points

        Returns:
            4-element tuple containing

            - **o_l** (*func*): outter curve of the lower lip
            - **o_u** (*func*): outter curve of the upper lip
            - **i_u** (*func*): inner curve of the upper lip
            - **i_l** (*func*): inner curve of the lower lip

        """
       
        points=np.array(points)

        # first 12 points indicate outter parts of the lips and the rest indicate inner parts.
        outter_x = np.array((points[:12][:, 0]))
        outter_y = np.array(points[:12][:, 1])
        inner_x = np.array(points[12:][:, 0])
        inner_y = np.array(points[12:][:, 1])

        up_left_end = 4
        up_right_end = 7
        in_left_end = 3
        in_right_end = 7

        lower_left_end = 5
        upper_left_end = 11
        lower_right_end = 16
        upper_right_end = 22

        o_l = self.__inter([outter_x[0]] + outter_x[up_right_end - 1:][::-1].tolist(),
                    [outter_y[0]] + outter_y[up_right_end - 1:][::-1].tolist(), 'cubic')
        o_u = self.__inter( outter_x[:up_right_end][::-1].tolist(),
                    outter_y[:up_right_end][::-1].tolist(), 'cubic')

        i_u = self.__inter( inner_x[:in_right_end][::-1].tolist(),
                    inner_y[:in_right_end][::-1].tolist(), 'cubic')
        i_l = self.__inter([inner_x[0]] + inner_x[in_right_end - 1:][::-1].tolist(),
                    [inner_y[0]] + inner_y[in_right_end - 1:][::-1].tolist(), 'cubic')

        return o_l, o_u, i_u, i_l, outter_x, inner_x






    def fill_lips(self, o_l, o_u, i_u, i_l, outter_x, inner_x):

        """  finds all the points fillig the lips
        Args:
            param1 : outter lower curve
            param2 : outter upper curve
            param3 : inner upper curve
            param4 : inner lower curve
            param5 : detected outter points of the lips
            param6 : detected inner points of the lips

        Returns:
            4-element tuple containing

            - **x** (*array*): X cordinates of the lips
            - **y** (*array*): Y cordinates of the lips
            - **x2** (*array*): X cordinates of the lower lip
            - **y2** (*array*): Y cordinates of the lower lip
              
        Todo:
            * should optimize the return vlues
            
        """
        x = []  
        y = []  
        x2 = []
        y2 = []
        for i in range(int(inner_x[0]),int(inner_x[6])):
            for j in range(int(o_u[0](i)),int(i_u[0](i))):
                x.append(j)
                y.append(i)
            for j in range(int(i_l[0](i)), int(o_l[0](i))):
                x.append(j)
                y.append(i)
                x2.append(j)
                y2.append(i)

        for i in range(int(outter_x[0]),int(inner_x[0])):
            for j in range(int(o_u[0](i)),int(o_l[0](i))):
                x.append(j)
                y.append(i)

        for i in range(int(inner_x[6]),int(outter_x[6])):
            for j in range(int(o_u[0](i)),int(o_l[0](i))):
                x.append(j)
                y.append(i)

        return x,y, x2,y2





    def change_rgb(self,x,y, r, g, b):
        """  changes  RGB color of the lips

        Args:
            param1 : X cordinates of the lips
            param2 : Y cordinates of the lips
            param3 : red 
            param4 : green
            param5 : blue 
        """


        mask=np.zeros([self.height,self.width],dtype='float64')
        mask[x,y] =255
        
       
        # converting lips part of the original image to LAB color space
        lip_LAB = color.rgb2lab((self.im_copy[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
        # print(lip_LAB.shape, mask.shape)

        L, A, B = mean(lip_LAB[:, 0]), mean(lip_LAB[:, 1]), mean(lip_LAB[:, 2])

        # meanLf = sum(sum(np.multiply(lip_LAB[:,:,0], mask))) / sum(mask.reshape(-1, 1))
        # converting the color of the lipstick to LAB
        L1, A1, B1 = color.rgb2lab(np.array((float(r) / 255.,float(g) / 255., float(b) / 255.)).reshape(1, 1, 3)).reshape(3, )

        G = L1 / L
        Op = self.intensity

        lip_LAB= lip_LAB.reshape(len(x), 1, 3)
        lip_LAB[:,:,1:3] = Op * np.array([A1,B1]) + (1-Op) * lip_LAB[:,:,1:3]
        # lip_LAb[:,:,1]= Op*A1+(1-Op) * lip_LAB[:,:,1]
        # lip_LAB[:,:,2]= Op*A1+(1-Op) * lip_LAB[:,:,2]
        lip_LAB[:,:,0] = lip_LAB[:,:,0] * (1 + Op * (G-1))


       



        # converting the lip parts back to RGB
        self.im_copy[x, y] = color.lab2rgb(lip_LAB).reshape(len(x), 3) * 255


    
    def __inter(self,lx, ly, k1='cubic'):
        unew = np.arange(lx[0], lx[-1] + 1, 1)
        f2 = interp1d(lx, ly, kind=k1)
        return f2, unew
       
    def moist(self, x,y, r, g, b):
        """  finds all the points fillig the lips
        Args:
            param1 : if motion is detected this parameter is the X cordinate of the lower lip otherwise the X cordinates of the previous frame gloss parts must be passed
            param2 : if motion is detected this parameter is the Y cordinate of the lower lip otherwise the Y cordinates of the previous frame gloss parts must be passed
            param3 : whether motion is detected
            param4 : red
            param5 : green
            param6 : blue

        Returns:
            42-element tuple containing

            -  (*array*): X cordinates of the pixels of the lips which must be glossy
            -  (*array*): Y cordinates of the pixels of the lips which must be glossy
              
        Todo:
            * needs major cleanup
            
        """


  

        val = color.rgb2lab((self.im_copy[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)


        L, A, B = mean(val[:, 0]), mean(val[:, 1]), mean(val[:, 2])
        L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )
        ll, aa, bb = L1 - L, A1 - A, B1 - B

        length = int(len(x)/6)
        Li = val[:, 0]
        light_points = sorted(Li)[-length:]
        min_val = min(light_points)
        max_val = max(light_points)


        index = []
        for i in range(len(val[:, 0])):
            if (val[i, 0] <= max_val and val[i, 0] >=min_val):
                val[i, 0]+= ll*self.intensitymoist
                index.append(i)
        
        r_img = (self.im_copy[x, y][:, 0]).flatten()

        # light_points = sorted(Li)[-100:]
        # min_val = min(light_points)
        # max_val = max(light_points)

        

        
   

        # height,width = self.image.shape[:2]
        # filter = np.zeros((height,width))
        # cv2.fillConvexPoly(filter,np.array(c_[ y, x],dtype = 'int32'),1)
        # filter = cv2.GaussianBlur(filter,(81,81),0)

        # # Erosion to reduce blur size
        # kernel = np.ones((20,20),np.uint8)
        # filter = cv2.erode(filter,kernel,iterations = 1)
        # alpha=np.zeros([height,width,3],dtype='float64')
        # alpha[:,:,0]=filter
        # alpha[:,:,1]=filter
        # alpha[:,:,2]=filter
        # self.im_copy = (alpha*self.image+(1-alpha)*self.im_copy).astype('uint8')


        # val[:, 0] +=ll*self.intensitymoist
        # val[:, 1] +=aa*self.intensitymoist
        # val[:, 2] += bb*self.intensitymoist
        
        self.im_copy[x, y] = color.lab2rgb(val).reshape(len(x), 3) * 255


        # print(min_val)

        # L, A, B = mean(val[:, 0]), mean(val[:, 1]), mean(val[:, 2])
        # L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )
        # ll, aa, bb = L1 - L, A1 - A, B1 - B
        # val[:, 0] +=ll*self.intensitymoist
        # val[:, 1] +=aa*self.intensitymoist
        # val[:, 2] += bb*self.intensitymoist
        # self.image[k1, f1] = color.lab2rgb(val.reshape(len(k1), 1, 3)).reshape(len(f1), 3) * 255


        # #guassian blur
        # height,width = self.image.shape[:2]
        filter = np.zeros((self.height,self.width))
        # cv2.fillConvexPoly(filter,np.array(c_[f1, k1],dtype = 'int32'),1)
        # filter = cv2.GaussianBlur(filter,(31,31),0)

        # # Erosion to reduce blur size
        kernel = np.ones((70,70),np.uint8)
        filter = cv2.erode(filter,kernel,iterations = 1)
        alpha=np.zeros([self.height,self.width,3],dtype='float64')
        alpha[:,:,0]=filter
        alpha[:,:,1]=filter
        alpha[:,:,2]=filter
        # self.im_copy = (alpha*self.image+(1-alpha)*self.im_copy).astype('uint8')
        return 


    
    def fill_soft(self, x , y):

        """  fills lip with soft color

        Args:
            param1 : X cordinates of the lips
            param2 : Y cordinates of the lips
            param3 : X cordinates of the lower lip
            param4 : Y cordinates of the lower lip  
        """

        #guassian blur
        height,width = self.image.shape[:2]
        filter = np.zeros((height,width))
        cv2.fillConvexPoly(filter,np.array(c_[y, x],dtype = 'int32'),1)
        filter = cv2.GaussianBlur(filter,(31,31),0)
        kernel = np.ones((10,10),np.uint8)
        filter = cv2.erode(filter,kernel,iterations = 1)

  
        alpha=np.zeros([height,width,3],dtype='float64')
        alpha[:,:,0]=filter
        alpha[:,:,1]=filter
        alpha[:,:,2]=filter

        
        mask = (alpha*self.im_copy+(1-alpha)*self.image).astype('uint8')
        cv2.imwrite('./data/mask.jpg',mask)
        self.im_copy = (alpha*self.im_copy+(1-alpha)*self.image).astype('uint8')


    

    def fill_solids(self,x,y):
        
        """  fills lip with soft color

        Args:
            param1 : X cordinates of the lips
            param2 : Y cordinates of the lips
            param3 : X cordinates of the lower lip
            param4 : Y cordinates of the lower lip  
        """

        #guassian blur
        height,width = self.image.shape[:2]
        filter = np.zeros((height,width))
        cv2.fillConvexPoly(filter,np.array(c_[y, x],dtype = 'int32'),1)
        filter = cv2.GaussianBlur(filter,(31,31),0)
        kernel = np.ones((5,5),np.uint8)
        filter = cv2.erode(filter,kernel,iterations = 1)

  
        alpha=np.zeros([height,width,3],dtype='float64')
        alpha[:,:,0]=filter
        alpha[:,:,1]=filter
        alpha[:,:,2]=filter

        
        mask = (alpha*self.im_copy+(1-alpha)*self.image).astype('uint8')
        cv2.imwrite('./data/mask.jpg',mask)
        self.im_copy = (alpha*self.im_copy+(1-alpha)*self.image).astype('uint8')

    


    def apply_lipstick(self,img, x,y, r, g, b, intensity, lipstick_type, gloss):
        """apllies lipstick on thedetected face

        Args:
            param1 : X cordinates 
            param2 : Y cordinates 
            param5 : red 
            param6 : green
            param7 : blue
            param9 : type of the lipstick which can be soft and hard
            param10 : whether gloss need to be added to the lipstick
            
        Returns: 
            the imagee of the face with lipstick applied   
        """
        self.image = img
        self.im_copy = img.copy()
        self.height, self.width = self.image.shape[:2]
        self.intensity = intensity
        points = self.get_lips(x, y)
        o_l, o_u, i_u, i_l, outter_x, inner_x =self.draw_curves(points)
        x , y , lowerx,lowery= self.fill_lips( o_l, o_u, i_u, i_l, outter_x, inner_x )
        if (gloss):
            print("Gloss")
            self.moist(lowerx, lowery, 230 , 230, 230)
        self.change_rgb(x,y, r, g, b)
        if(lipstick_type == "hard"):
            self.fill_solids(x,y)
        elif(lipstick_type == "soft"):
            self.fill_soft(x,y)

        self.x_all = x
        self.y_all = y



        return self.im_copy
