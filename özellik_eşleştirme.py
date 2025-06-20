import cv2
import numpy as np

class NesneAlgila():
    def __init__ (self):
        self.img1 = cv2.imread("kitap.jpg", 0)
        self.img2 = cv2.imread("kitap_2.jpg", 0)

    def sift(self):
        sift = cv2.SIFT.create()
        self.kp1, self.des1 = sift.detectAndCompute(self.img1, None)
        self.kp2, self.des2 = sift.detectAndCompute(self.img2, None)
        self.flann()

    def flann(self):
        index_params = dict(algorithm = 1, trees = 5)
        search_params = dict(checks = 200)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.matches = flann.knnMatch(self.des1, self.des2, k=2)
        self.kontrol()

    def kontrol(self):
        self.good = []
        for m,n in self.matches:
            if m.distance < 0.7 * n.distance:
                self.good.append(m)
        self.nesne_bul()

    def nesne_bul(self):
        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in self.good]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in self.good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
        self.matchesMask = mask.ravel().tolist()

        h,w = self.img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        self.img2 = cv2.polylines(self.img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        self.cizim()

    def cizim(self):
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = self.matchesMask, # draw only inliers
                   flags = 2)
 
        img3 = cv2.drawMatches(self.img1,self.kp1,self.img2,self.kp2,self.good,None,**draw_params)
        cv2.imshow("img", img3)
        cv2.imwrite("output.jpg", img3)

if __name__ == "__main__":
    baslat = NesneAlgila()
    baslat.sift()

    cv2.waitKey(0)
    cv2.destroyAllWindows()



