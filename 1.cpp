#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <cmath>
#include <sstream>
#include <vector>

using namespace cv;
using namespace std;
using namespace chrono;

// ------------------------------------------------------------
// 直方图分位点（灰度）
// ------------------------------------------------------------
static int percentile(const cv::Mat& g8, double p) {
    CV_Assert(g8.type() == CV_8U && p >= 0.0 && p <= 1.0);
    int hist[256] = { 0 };
    for (int y = 0; y < g8.rows; ++y) {
        const uchar* r = g8.ptr<uchar>(y);
        for (int x = 0; x < g8.cols; ++x) hist[r[x]]++;
    }
    long total = (long)g8.total();
    long target = (long)std::round(p * total);
    long c = 0;
    for (int i = 0; i < 256; ++i) { c += hist[i]; if (c >= target) return i; }
    return 255;
}

// ------------------------------------------------------------
// 三段式 LUT
// ------------------------------------------------------------
static cv::Mat buildHighlightLUT(int m, int h,
    double gamma_mid = 1.4,
    double gamma_hi = 0.7,
    double boost_hi = 1.10) {
    m = std::clamp(m, 0, 254);
    h = std::clamp(h, m + 1, 255);
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        double x = i / 255.0;
        double y = x;
        if (i < m) {
            y = x;
        }
        else if (i < h) {
            double xm = (x - (m / 255.0)) / ((h - m) / 255.0);
            xm = std::clamp(xm, 0.0, 1.0);
            double ym = std::pow(xm, gamma_mid);
            y = (m / 255.0) + ym * ((h - m) / 255.0);
        }
        else {
            double xh = (x - (h / 255.0)) / ((255 - h) / 255.0);
            xh = std::clamp(xh, 0.0, 1.0);
            double yh = std::pow(xh, gamma_hi);
            y = (h / 255.0) + yh * ((255 - h) / 255.0);
            y = std::min(1.0, y * boost_hi);
        }
        lut.at<uchar>(i) = (uchar)std::round(std::clamp(y, 0.0, 1.0) * 255.0);
    }
    return lut;
}

// ------------------------------------------------------------
// 从 start 沿单位方向 dir 在二值图 bin 上扫描，直到命中白色像素(255)
// ------------------------------------------------------------
static bool scanUntilWhiteBinary(const cv::Mat& bin, cv::Point2f start, cv::Point2f dir, cv::Point& hit) {
    float n = std::hypot(dir.x, dir.y);
    if (n < 1e-6f) return false;
    dir.x /= n; dir.y /= n;

    cv::Point2f p = start;
    cv::Point last(-9999, -9999);
    const int maxSteps = (int)std::ceil(1.5 * std::max(bin.cols, bin.rows));
    for (int s = 1; s <= maxSteps; ++s) {
        p += dir;
        int xi = cvRound(p.x), yi = cvRound(p.y);
        if (xi < 0 || yi < 0 || xi >= bin.cols || yi >= bin.rows) break;
        if (xi == last.x && yi == last.y) continue;
        last = { xi, yi };
        if (bin.at<uchar>(yi, xi) == 255) { hit = last; return true; }
    }
    return false;
}

int main() {
    // ------------------ 开关与参数 ------------------
    bool SAVE_DATA = true;                 // 保存数据
    bool SHOW_VISUALIZATION = true;        // 显示可视化窗口
    bool ENABLE_PERSPECTIVE_CORRECTION = true; // 透视校正
    bool USE_ENHANCED_PREPROCESS = true;   // 预处理增强（双边+CLAHE+LUT+Otsu）
    const double scaleFactor = 0.5;        // 计算降采样倍率

    bool ENABLE_COOLING_RATE = true;       // 计算冷却速率
    double scanSpeed_mmps = 3.0;           // 扫描速度 mm/s

    // ------------------ 相机参数 ------------------
    Mat K = (Mat_<double>(3, 3) <<
        19376.9754, 0.0, 708.691041,
        0.0, 19397.0337, 542.250680,
        0.0, 0.0, 1.0);

    Mat rvec = (Mat_<double>(3, 1) <<
        0.0, -0.1745329252, 0.0);

    Mat tvec = (Mat_<double>(3, 1) <<
        -0.25589408, 0, -1.45240031);

    Mat t_mm = tvec * 1000.0;
    Mat n = (Mat_<double>(3, 1) << 0, 0, 1);
    Mat R;
    Rodrigues(rvec, R);

    setUseOptimized(true);
    setNumThreads(getNumberOfCPUs());

    if (SHOW_VISUALIZATION) {
        namedWindow("Corrected", WINDOW_NORMAL);
        namedWindow("Binary", WINDOW_NORMAL);
        namedWindow("Corrected + Contour", WINDOW_NORMAL);
        namedWindow("Tone", WINDOW_NORMAL);
    }

    // ---------- 运行级输出目录：output/时间戳 ----------
    string runFolder;
    if (SAVE_DATA) {
        auto timestamp = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
#ifdef _WIN32
        system("mkdir output >nul 2>nul");
        system(("mkdir \"output\\" + to_string(timestamp) + "\"").c_str());
        runFolder = "output/" + to_string(timestamp);
#else
        system("mkdir -p output");
        system(("mkdir -p output/" + to_string(timestamp)).c_str());
        runFolder = "output/" + to_string(timestamp);
#endif
        cout << "Saving data to: " << runFolder << endl;
    }

    // ============ 扫描文件夹内所有 .tiff / .tif ============
    vector<cv::String> fileNames;
    cv::glob("D:/Project2/1/*.tiff", fileNames, false);
    vector<cv::String> tifNames;
    cv::glob("D:/Project2/1/*.tif", tifNames, false);
    fileNames.insert(fileNames.end(), tifNames.begin(), tifNames.end());
    sort(fileNames.begin(), fileNames.end());
    fileNames.erase(unique(fileNames.begin(), fileNames.end()), fileNames.end());

    if (fileNames.empty()) {
        cerr << "未在 D:/Project2/1/ 找到 .tiff 或 .tif 图像文件！" << endl;
        return 1;
    }
    else {
        cout << "共找到 " << fileNames.size() << " 个图像文件，将逐一处理..." << endl;
    }

    int frame_id = 0;

    for (size_t fi = 0; fi < fileNames.size(); ++fi) {
        const string imagePath = fileNames[fi];
        auto t0 = steady_clock::now();

        // ====== 读取灰度图 ======
        Mat img = imread(imagePath, IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "无法加载图像: " << imagePath << endl;
            continue;
        }

        // ------------------ 透视校正（灰度） ------------------
        Mat corrected;
        if (ENABLE_PERSPECTIVE_CORRECTION) {
            double d = t_mm.dot(n);
            if (abs(d) < 1e-6) {
                cerr << "d 值异常，跳过：" << imagePath << endl;
                continue;
            }
            Mat K_inv; invert(K, K_inv, DECOMP_SVD);
            Mat H = K * (R - t_mm * n.t() / d) * K_inv;
            Mat H_inv; invert(H, H_inv);
            Mat H_final = H_inv;

            // 自适应视图
            Size dstSize = img.size();
            vector<Point2f> srcCorners = {
                Point2f(0.f, 0.f),
                Point2f((float)img.cols - 1, 0.f),
                Point2f((float)img.cols - 1, (float)img.rows - 1),
                Point2f(0.f, (float)img.rows - 1)
            };
            vector<Point2f> warpedCorners;
            perspectiveTransform(srcCorners, warpedCorners, H_final);
            float minX = numeric_limits<float>::infinity();
            float maxX = -numeric_limits<float>::infinity();
            float minY = numeric_limits<float>::infinity();
            float maxY = -numeric_limits<float>::infinity();
            for (const auto& p : warpedCorners) {
                minX = std::min(minX, p.x);  maxX = std::max(maxX, p.x);
                minY = std::min(minY, p.y);  maxY = std::max(maxY, p.y);
            }
            double bboxW = std::max(1.0f, maxX - minX);
            double bboxH = std::max(1.0f, maxY - minY);
            double s = std::min(dstSize.width / bboxW, dstSize.height / bboxH);

            Mat T_toOrigin = (Mat_<double>(3, 3) << 1, 0, -minX, 0, 1, -minY, 0, 0, 1);
            Mat S = (Mat_<double>(3, 3) << s, 0, 0, 0, s, 0, 0, 0, 1);
            double fitW = bboxW * s, fitH = bboxH * s;
            double tx = (dstSize.width - fitW) * 0.5, ty = (dstSize.height - fitH) * 0.5;
            Mat C_center = (Mat_<double>(3, 3) << 1, 0, tx, 0, 1, ty, 0, 0, 1);
            Mat H_view = C_center * S * T_toOrigin * H_final;

            warpPerspective(img, corrected, H_view, dstSize,
                INTER_LINEAR, BORDER_CONSTANT, Scalar(0)); // 灰度背景 0
        }
        else {
            corrected = img.clone();
        }

        // ------------------ 可选降采样（灰度） ------------------
        if (scaleFactor != 1.0) {
            resize(corrected, corrected, Size(), scaleFactor, scaleFactor, INTER_AREA);
        }

        // ------------------ 灰度直接用作后续处理 ------------------
        Mat gray = corrected.clone();
        Mat tone = gray.clone();
        Mat binary;

        if (USE_ENHANCED_PREPROCESS) {
            Mat den; bilateralFilter(gray, den, 9, 50, 7);
            Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
            Mat cla; clahe->apply(den, cla);

            int m = percentile(cla, 0.50);
            int h = percentile(cla, 0.95);
            Mat lut = buildHighlightLUT(m, h, 1.4, 0.7, 1.10);
            LUT(cla, lut, tone);

            threshold(tone, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
        }
        else {
            threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
        }

        // ------------------ 轮廓与外接最小矩形 ------------------
        vector<vector<Point>> contours;
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
            cerr << "未找到轮廓，文件：" << imagePath << endl;
            if (SHOW_VISUALIZATION) {
                imshow("Corrected", corrected); // 灰度
                imshow("Binary", binary);
                imshow("Tone", tone);
                int key = waitKey(1);
                if (key == 27) break; // ESC 退出
            }
            continue;
        }

        size_t largestContourIdx = 0;
        double maxArea = 0.0;
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = contourArea(contours[i]);
            if (area > maxArea) { maxArea = area; largestContourIdx = i; }
        }

        RotatedRect minRect = minAreaRect(contours[largestContourIdx]);
        float rectWidth = minRect.size.width;
        float rectHeight = minRect.size.height;
        float widthInPixels = std::min(rectWidth, rectHeight);

        // px→mm
        double widthInMM = (0.0078 / scaleFactor) * (widthInPixels - 20);

        // ------------------ 冷却速率 ------------------
        bool haveCooling = false;
        double coolingRate = std::numeric_limits<double>::quiet_NaN();

        if (ENABLE_COOLING_RATE) {
            float angle_deg = minRect.angle;
            bool widthIsLong = (rectWidth >= rectHeight);
            float theta_deg = angle_deg + (widthIsLong ? 0.0f : 90.0f);
            float theta = theta_deg * (float)CV_PI / 180.0f;
            Point2f u_vec(std::cos(theta), std::sin(theta));

            Point2f rp[4]; minRect.points(rp);
            vector<pair<Point2f, Point2f>> edges = { {rp[0],rp[1]}, {rp[1],rp[2]}, {rp[2],rp[3]}, {rp[3],rp[0]} };
            vector<double> lens(4);
            for (int i = 0; i < 4; ++i) lens[i] = norm(edges[i].second - edges[i].first);

            vector<int> idx = { 0,1,2,3 };
            sort(idx.begin(), idx.end(), [&](int a, int b) { return lens[a] < lens[b]; });

            pair<Point2f, Point2f> shortEdge1 = edges[idx[0]];
            pair<Point2f, Point2f> shortEdge2 = edges[idx[1]];
            Point2f mid1f = 0.5f * (shortEdge1.first + shortEdge1.second);
            Point2f mid2f = 0.5f * (shortEdge2.first + shortEdge2.second);
            Point mid1(cvRound(mid1f.x), cvRound(mid1f.y));
            Point mid2(cvRound(mid2f.x), cvRound(mid2f.y));

            auto in_img_bounds = [&](int x, int y) { return !(x < 0 || y < 0 || x >= gray.cols || y >= gray.rows); };

            if (!in_img_bounds(mid1.x, mid1.y) || !in_img_bounds(mid2.x, mid2.y)) {
                cerr << "短边中点越界，文件：" << imagePath << "\n";
            }
            else {
                Mat binary220;
                threshold(gray, binary220, 230, 255, THRESH_BINARY);

                auto traceAlongU = [&](const Point2f& startf, Point& endPt, double& len_px)->bool {
                    Point hit;
                    bool ok = scanUntilWhiteBinary(binary220, startf, u_vec, hit);
                    if (!ok) ok = scanUntilWhiteBinary(binary220, startf, -u_vec, hit);
                    if (!ok) return false;
                    endPt = hit;
                    len_px = norm(endPt - Point(cvRound(startf.x), cvRound(startf.y)));
                    return len_px > 0.0;
                    };

                Point q1, q2; double d1 = 0.0, d2 = 0.0;
                bool ok1 = traceAlongU(mid1f, q1, d1);
                bool ok2 = traceAlongU(mid2f, q2, d2);

                if (!ok1 && !ok2) {
                    cerr << "冷却速率计算失败：沿长边方向未命中高亮，文件：" << imagePath << endl;
                }
                else {
                    Point Pstart, Pend;
                    double dx_px;
                    if (ok1 && (!ok2 || d1 >= d2)) {
                        Pstart = mid1; Pend = q1; dx_px = d1;
                    }
                    else {
                        Pstart = mid2; Pend = q2; dx_px = d2;
                    }

                    float T1 = gray.at<uchar>(Pstart);
                    float T2 = gray.at<uchar>(Pend);
                    // 注意：这里沿用你原代码的 0.0077/scaleFactor
                    double dx_mm = dx_px * (0.0077 / scaleFactor);
                    double dT = (double)T1 - (double)T2;
                    coolingRate = (dx_mm > 1e-9) ? (dT * scanSpeed_mmps) / dx_mm
                        : std::numeric_limits<double>::quiet_NaN();

                    haveCooling = !std::isnan(coolingRate);
                }
            }
        }

        auto t1 = steady_clock::now();
        auto duration_ms = duration_cast<milliseconds>(t1 - t0).count();

        cout << "----------------------------------------\n";
        cout << "File: " << imagePath << "\n";
        cout << "Frame ID: " << frame_id << "\n";
        cout << "Meltpool Width: " << widthInPixels << " px ≈ "
            << fixed << setprecision(2) << widthInMM << " mm" << endl;
        if (ENABLE_COOLING_RATE) {
            if (haveCooling) cout << "CoolingRate: " << fixed << setprecision(2) << coolingRate << " GL/s\n";
            else             cout << "CoolingRate: N/A\n";
        }
        cout << "Processing Time: " << duration_ms << " ms" << endl;

        // ------------------ 可视化：把灰度转 BGR，再画轮廓/矩形/文字 ------------------
        Mat correctedDisplay;
        cvtColor(corrected, correctedDisplay, COLOR_GRAY2BGR); // 仅用于彩色叠加显示

        drawContours(correctedDisplay, contours, static_cast<int>(largestContourIdx), Scalar(0, 0, 255), 2);
        Point2f rect_points[4];
        minRect.points(rect_points);
        for (int j = 0; j < 4; j++)
            line(correctedDisplay, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 255, 0), 2);

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "Width: " << widthInMM << " mm | Cooling: ";
        if (ENABLE_COOLING_RATE && haveCooling) ss << coolingRate << " GL/s";
        else ss << "N/A";
        putText(correctedDisplay, ss.str(), Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

        if (SHOW_VISUALIZATION) {
            imshow("Corrected", corrected);              
            imshow("Binary", binary);
            imshow("Corrected + Contour", correctedDisplay); 
        }

        // ------------------ 保存数据（每帧一组文件） ------------------
        if (SAVE_DATA) {
            std::ostringstream nameBase;
            nameBase << std::setw(6) << std::setfill('0') << frame_id;

            imwrite(runFolder + "/" + nameBase.str() + "_corrected.png", corrected);
            imwrite(runFolder + "/" + nameBase.str() + "_binary.png", binary);
            imwrite(runFolder + "/" + nameBase.str() + "_corrected_contour.png", correctedDisplay);

            ofstream log(runFolder + "/" + nameBase.str() + "_data.txt");
            log << std::fixed << std::setprecision(2);
            log << "File: " << imagePath << "\n";
            log << "Frame_ID: " << frame_id << "\n";
            log << "Width_px: " << widthInPixels << "\n";
            log << "Width_mm: " << widthInMM << "\n";
            if (ENABLE_COOLING_RATE && haveCooling) {
                log << "CoolingRate_GLps: " << coolingRate << "\n";
            }
            else {
                log << "CoolingRate_GLps: N/A\n";
            }
            log << "ProcessingTime_ms: " << duration_ms << "\n";
            log.close();
        }

        int key = SHOW_VISUALIZATION ? waitKey(1) : -1;
        if (key == 27) break;  // ESC 退出

        frame_id++;
        // （for 循环自动进入下一个文件）
    }

    destroyAllWindows();
    return 0;
}


