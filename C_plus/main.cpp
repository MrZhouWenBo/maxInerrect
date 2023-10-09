#include <iostream>
#include <vector>
#include <stack>
#include"opencv2/opencv.hpp"

namespace maxrectTest{

    std::vector<int> largestRectangleArea(int* heights, int dataLenght) {
        std::vector<int> stack;
        int left = 0;
        int right = 0;
        int height = 0;
        int res = 0;
        int i = 0;

        while (i < dataLenght) {
            if (stack.empty() || heights[i] >= heights[stack.back()]) {
                stack.push_back(i);
                i++;
            } else {
                int k = stack.back();
                stack.pop_back();
                int area = heights[k] * ((i - (stack.empty() ? 0 : stack.back() + 1)));
                if (area > res) {
                    left = (stack.empty() ? 0 : stack.back());
                    right = i;
                    height = heights[k];
                }
                res = std::max(res, area);
            }
        }

        while (!stack.empty()) {
            int k = stack.back();
            stack.pop_back();
            int area = heights[k] * ((i - (stack.empty() ? 0 : stack.back() + 1)));
            if (area > res) {
                left = (stack.empty() ? 0 : stack.back());
                right = i;
                height = heights[k];
            }
            res = std::max(res, area);
        }

        return {res, left, right, height};
    }

    std::vector<int> getMaxrect_xywh(const cv::Mat& dataMatrix) {
        std::vector<int> retTect;
        int num_rows_ys = dataMatrix.rows;
        int num_cols_xs = dataMatrix.cols;
        
        cv::Mat colSumDatas = cv::Mat::zeros(dataMatrix.size(), CV_32SC1);
        int maxArea = 0;
        
        for (int c = 0; c < num_cols_xs; c++) {
            uchar tmp = dataMatrix.data[c];
            if (tmp > 0) {
                ((int*)colSumDatas.data)[c] = 1;
            } else {
                ((int*)colSumDatas.data)[c] = 0;
            }
        }

        for (int r = 1; r < num_rows_ys; r++) {
            uchar *tmpDataPtr = dataMatrix.data +  r * num_cols_xs;
            int* binaryLastLineDataPtr = (int*)colSumDatas.data + (r-1)*num_cols_xs;
            int* binaryCurrLineDataPtr = (int*)colSumDatas.data + (r-0)*num_cols_xs;
            for (int c = 0; c < num_cols_xs; c++) {
                uchar tmp = tmpDataPtr[c];
                if (tmp > 0) {
                    binaryCurrLineDataPtr[c] = binaryLastLineDataPtr[c] + 1;
                } else {
                    binaryCurrLineDataPtr[c] = 0;
                }
            }
        }

        for (int r = 0; r < num_rows_ys; r++) {
            int* tmpRowDataPtr = (int*)colSumDatas.data + r * num_cols_xs;
            std::vector<int> result = largestRectangleArea(tmpRowDataPtr, num_cols_xs);
            int res = result[0];
            int left = result[1];
            int right = result[2];
            int height = result[3];

            if (res > maxArea) {
                maxArea = res;
                retTect = {left, r - height, right - left, height};
            }
        }

        std::cout << "retTect: ";
        for (int i = 0; i < retTect.size(); i++) {
            std::cout << retTect[i] << " ";
        }
        std::cout << std::endl;

        return retTect;
    }
        
    
    void test()
    {
        std::string inImgPath = "../datas/binaryImg.png";
        std::string tmpSavePath = "../datas/tmp.png";
        std::string resultSavePath = "../datas/result.png";
        cv::Mat colorImg = cv::imread(inImgPath);
        cv::Mat grayImg;
        cv::cvtColor(colorImg, grayImg, cv::COLOR_BGR2GRAY);
        std::vector<std::vector<cv::Point>> contours; 
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(grayImg, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        float contourMeanLen = 0;
        for (int i = 0; i < contours.size(); i++)
        {
            contourMeanLen += contours[i].size();
        }
        contourMeanLen /= contours.size();
        // 逐个获取外接矩形 然后求其内部窗户区域的最大内接矩形
        for (int i = 0; i < contours.size(); i++)
        {
            // 筛选contour
            if (contours[i].size() < (contourMeanLen / 10))
            {
                continue;
            }
            cv::Rect rectROI = cv::boundingRect(cv::Mat(contours[i]));
            // 裁剪出窗户外接矩形的mat数据
            cv::Mat roiBinaryMat = grayImg(rectROI);
            cv::imwrite(tmpSavePath, roiBinaryMat);
            cv::Mat roiBinaryMat_int;
            roiBinaryMat.convertTo(roiBinaryMat_int, CV_8UC1);  //CV_32SC1
            // 获取该裁剪矩形内窗户区域的最大内接矩形
            std::vector<int> result =  getMaxrect_xywh(roiBinaryMat_int);
            cv::Point start(result[0] + rectROI.x, result[1] + rectROI.y);
            cv::Point end(result[0] + rectROI.x + result[2], result[1] + rectROI.y + result[3]);
            cv::rectangle(colorImg, start,  end, cv::Scalar(0, 0, 255), 2);
            
        }
        cv::imwrite(resultSavePath, colorImg);
    }

    int test_largestRectangleArea() {
        std::vector<int> heights = {2,2, 2,1,2, 5,4,6};
        std::vector<int> result = largestRectangleArea(&heights[0], heights.size());
        int res = result[0];
        int left = result[1];
        int right = result[2];
        int height = result[3];

        std::cout << "Largest rectangle area: " << res << std::endl;
        std::cout << "Left index: " << left << std::endl;
        std::cout << "Right index: " << right << std::endl;
        std::cout << "Height: " << height << std::endl;

        return 0;
    }

}

int main(int argc, char *argv[]) {
    
    maxrectTest::test();
    maxrectTest::test_largestRectangleArea();
    return 0;
}