import numpy as np
import cv2
import os



def getMaxrect(dataMatrix):
    '''
        dataMatrix 输入矩形 0 1 数据
        return     返回相对矩形数据的最大 最大内接正方形位置信息
    '''
    retTect= None
    num_rows_ys = dataMatrix.shape[0]
    num_cols_xs = dataMatrix.shape[1]
    dpDatas = np.zeros([dataMatrix.shape[0], dataMatrix.shape[1]], np.int)
    maxLength = 0
    # 第一列赋值
    for r in range(0, num_rows_ys):
        if(dataMatrix[r][0] > 0):
            dpDatas[r][0] = 1
        else:
            dpDatas[r][0] = 0
        maxLength=max(maxLength,dpDatas[r][0])
    # 第一行赋值
    for c in range(0, num_cols_xs):
        if(dataMatrix[0][c] > 0):
            dpDatas[0][c] = 1
        else:
            dpDatas[0][c] = 0
        maxLength=max(maxLength,dpDatas[0][c])
    # 动态规划判断状态并更新动态规划内的数据
    for r in range(1, num_rows_ys):
         for c in range(1, num_cols_xs):
              if(dataMatrix[r][c] > 0):
                   dpDatas[r][c] = min(min(dpDatas[r-1][c], dpDatas[r][c -1]), dpDatas[r-1][c-1]) + 1
                   if (dpDatas[r][c] > maxLength):
                        retTect = (c - maxLength, r - maxLength, maxLength, maxLength)
                   maxLength = max(dpDatas[r][c], maxLength)
                   
    print(dpDatas.shape)
    return retTect


  


def largestRectangleArea(heights):
    """
    :type heights: List[int]
    :rtype: int
    """
    stack = list()
    left = 0
    right = 0
    height = 0
    res, i = 0, 0
    while i < len(heights):
        if not stack or (heights[i] >= heights[stack[-1]]):
            stack.append(i)
            i += 1
        else:
            k = stack.pop()
            area = heights[k]*((i - stack[-1] - 1) if stack else i)
            if area > res:
                left = (stack[-1] if stack else 0)
                right = i
                height = heights[k]

            res = max(res, heights[k]*((i - stack[-1] - 1) if stack else i))    
    while stack:
        k = stack.pop()
        area = heights[k]*((i - stack[-1] - 1) if stack else i)
        if area > res:
            left = (stack[-1] if stack else 0)
            right = i
            height = heights[k]
        res = max(res, heights[k]*((i - stack[-1] - 1) if stack else i))

    return res, left, right, height
  
def getMaxrect_xywh(dataMatrix):
    '''
        dataMatrix 输入矩形 0 1 数据

        return     返回相对矩形数据的最大 最大内接矩形位置信息
    '''
    retTect= None
    num_rows_ys = dataMatrix.shape[0]
    num_cols_xs = dataMatrix.shape[1]
    colSumDatas = np.zeros(dataMatrix.shape)
    maxArea = 0
    # 第一行赋值
    for c in range(0, num_cols_xs):
        if(dataMatrix[0][c] > 0):
            colSumDatas[0][c] = 1
        else:
            colSumDatas[0][c] = 0
    # 剩余行赋值
    for r in range(1, num_rows_ys):
        for c in range(0, num_cols_xs):
            if dataMatrix[r][c] > 0:
                colSumDatas[r][c] = colSumDatas[r-1][c] + 1
            else:
                colSumDatas[r][c] = 0
    # 逐行使用dp算法求最大面积
    for r in range(0, num_rows_ys):
        heights_datas = colSumDatas[r][:]
        
        res, left, right, height = largestRectangleArea(heights_datas)
        if(res > maxArea):
            maxArea = res
            retTect = [left, r - height, right-left, height]
    print("sdsdsdsd", retTect)
    return np.array(retTect).astype(dtype=int)

def maxInerRect(imggray, bgr_image):
    contours, hierarchy = cv2.findContours(imggray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 滤除过小的contours
    mean = 0
    for cnt in contours:
         mean = mean + cnt.shape[0]
    mean = mean / len(contours) / 10
    for cnt in contours:
        if cnt.shape[0] < mean:
             continue
        
        color = (0, 255, 0)  # 绿色
        thickness = 2
        
        # 获取到区域的外接矩形左上角坐标和宽高
        x, y, w, h = cv2.boundingRect(cnt)
        
        matrixImg = imggray[ y : y + h, x : x + w]
        rect = getMaxrect_xywh(matrixImg)
        cv2.rectangle(bgr_image, (rect[0] + x, rect[1] + y) , (rect[0] + rect[2] + x, rect[1] + rect[3] + y), color, thickness)
        
        
    return bgr_image

if __name__ == '__main__':
        testName = '../datas/binaryImg.png'
        saveRoot = '../datas/pythonret'
        img_gray = cv2.imread(testName, cv2.IMREAD_GRAYSCALE)
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        image = maxInerRect(img_gray, img_bgr)
        cv2.imwrite(os.path.join(saveRoot,testName.split('/')[-1]), image) 
