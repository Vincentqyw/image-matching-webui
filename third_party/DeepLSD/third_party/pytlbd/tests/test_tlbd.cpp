#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
#include <random>
#include "LineBandDescriptor.h"
#include <PairwiseLineMatching.h>
#include <EDLineDetector.h>
#include <gtest/gtest.h>

using namespace eth;

void draw_matches(const cv::Mat &cvLeftImage, const cv::Mat &cvRightImage,
                  const eth::ScaleLines &linesInLeft, const eth::ScaleLines &linesInRight,
                  const std::vector<std::pair<uint32_t, uint32_t>> &matchResult) {
  cv::Point startPoint;
  cv::Point endPoint;

  cv::Mat cvLeftColorImage, cvRightColorImage;
  cv::cvtColor(cvLeftImage, cvLeftColorImage, cv::COLOR_GRAY2BGR);
  cv::cvtColor(cvRightImage, cvRightColorImage, cv::COLOR_GRAY2BGR);

  int w = cvLeftImage.cols, h = cvLeftImage.rows;
  int lowest = 100, highest = 255;
  int range = (highest - lowest) + 1;
  unsigned int r, g, b; //the color of lines
  for (auto &lines_vec : linesInLeft) {
    r = lowest + int(rand() % range);
    g = lowest + int(rand() % range);
    b = lowest + int(rand() % range);
    startPoint = cv::Point(int(lines_vec[0].startPointX), int(lines_vec[0].startPointY));
    endPoint = cv::Point(int(lines_vec[0].endPointX), int(lines_vec[0].endPointY));
    cv::line(cvLeftColorImage, startPoint, endPoint, CV_RGB(r, g, b));
  }
  cv::imshow("Left", cvLeftColorImage);

  for (auto &lines_vec : linesInRight) {
    r = lowest + int(rand() % range);
    g = lowest + int(rand() % range);
    b = lowest + int(rand() % range);
    startPoint = cv::Point(int(lines_vec[0].startPointX), int(lines_vec[0].startPointY));
    endPoint = cv::Point(int(lines_vec[0].endPointX), int(lines_vec[0].endPointY));
    cv::line(cvRightColorImage, startPoint, endPoint, CV_RGB(r, g, b));
  }
  cv::imshow("Right", cvRightColorImage);

  ///////////####################################################################

  //store the matching results of the first and second images into a single image
  int lineIDLeft, lineIDRight;
  cv::cvtColor(cvLeftImage, cvLeftColorImage, cv::COLOR_GRAY2RGB);
  cv::cvtColor(cvRightImage, cvRightColorImage, cv::COLOR_GRAY2RGB);
  int lowest1 = 0, highest1 = 255;
  int range1 = (highest1 - lowest1) + 1;
  std::vector<unsigned int> r1(matchResult.size() / 2), g1(matchResult.size() / 2),
      b1(matchResult.size() / 2); //the color of lines
  for (unsigned int pair = 0; pair < matchResult.size() / 2; pair++) {
    r1[pair] = lowest1 + int(rand() % range1);
    g1[pair] = lowest1 + int(rand() % range1);
    b1[pair] = 255 - r1[pair];
    lineIDLeft = matchResult[pair].first;
    lineIDRight = matchResult[pair].second;
    startPoint.x = linesInLeft[lineIDLeft][0].startPointX;
    startPoint.y = linesInLeft[lineIDLeft][0].startPointY;
    endPoint.x = linesInLeft[lineIDLeft][0].endPointX;
    endPoint.y = linesInLeft[lineIDLeft][0].endPointY;
    cv::line(cvLeftColorImage,
             startPoint,
             endPoint,
             CV_RGB(r1[pair], g1[pair], b1[pair]),
             4,
             cv::LINE_AA);
    startPoint.x = linesInRight[lineIDRight][0].startPointX;
    startPoint.y = linesInRight[lineIDRight][0].startPointY;
    endPoint.x = linesInRight[lineIDRight][0].endPointX;
    endPoint.y = linesInRight[lineIDRight][0].endPointY;

    cv::line(cvRightColorImage,
             startPoint,
             endPoint,
             CV_RGB(r1[pair], g1[pair], b1[pair]),
             4,
             cv::LINE_AA);
  }

  cv::Mat cvResultColorImage1(h, w * 2, CV_8UC3);
  cv::Mat cvResultColorImage2, cvResultColorImage;

  cv::Mat out1 = cvResultColorImage1(cv::Rect(0, 0, w, h));
  cvLeftColorImage.copyTo(out1);
  cv::Mat out2 = cvResultColorImage1(cv::Rect(w, 0, w, h));
  cvRightColorImage.copyTo(out2);

  cvResultColorImage2 = cvResultColorImage1.clone();
  for (unsigned int pair = 0; pair < matchResult.size() / 2; pair++) {
    lineIDLeft = matchResult[pair].first;
    lineIDRight = matchResult[pair].second;
    startPoint.x = linesInLeft[lineIDLeft][0].startPointX;
    startPoint.y = linesInLeft[lineIDLeft][0].startPointY;
    endPoint.x = linesInRight[lineIDRight][0].startPointX + w;
    endPoint.y = linesInRight[lineIDRight][0].startPointY;
    cv::line(cvResultColorImage2,
             startPoint,
             endPoint,
             CV_RGB(r1[pair], g1[pair], b1[pair]),
             2,
             cv::LINE_AA);
  }
  cv::addWeighted(cvResultColorImage1, 0.5, cvResultColorImage2, 0.5, 0.0, cvResultColorImage);

  std::cout << "number of total matches = " << matchResult.size() / 2 << std::endl;
  cv::imshow("LBDSG", cvResultColorImage);
  cv::waitKey();
}

TEST(TLBD, line_matching_example_leuven) {

  //load first image from file
  cv::Mat cvLeftImage = cv::imread("../resources/leuven1.jpg", cv::IMREAD_GRAYSCALE);
  cv::Mat cvRightImage = cv::imread("../resources/leuven2.jpg", cv::IMREAD_GRAYSCALE);

  // 2.1. Detecting lines in the scale space
  eth::MultiOctaveSegmentDetector detector(std::make_shared<eth::EDLineDetector>());
  eth::ScaleLines linesInLeft = detector.octaveKeyLines(cvLeftImage);
  eth::ScaleLines linesInRight = detector.octaveKeyLines(cvRightImage);

  // 2.2. The band representation of the line support region & 2.3. The construction of the Line Band Descriptor
  eth::LineBandDescriptor lineDesc;
  std::vector<std::vector<cv::Mat>> descriptorsLeft, descriptorsRight;
  lineDesc.compute(cvLeftImage, linesInLeft, descriptorsLeft);
  lineDesc.compute(cvRightImage, linesInRight, descriptorsRight);

  // 3. Graph matching using spectral technique
  std::vector<std::pair<uint32_t, uint32_t>> matchResult;
  eth::PairwiseLineMatching lineMatch;
  lineMatch.matchLines(linesInLeft, linesInRight, descriptorsLeft, descriptorsRight, matchResult);

  // Show the result
//  draw_matches(cvLeftImage, cvRightImage, linesInLeft, linesInRight, matchResult);

  std::vector<std::pair<uint32_t, uint32_t>> expectedMatch = {
      {45, 60}, {44, 49}, {47, 62}, {162, 182}, {80, 91}, {0, 1}, {3, 4}, {79, 90}, {180, 54},
      {173, 193}, {125, 134},
      {2, 3}, {9, 18}, {352, 431}, {128, 145}, {24, 0}, {8, 17}, {272, 324}, {130, 146}, {146, 166},
      {48, 65},
      {245, 289}, {83, 92}, {390, 295}, {249, 296}, {15, 16}, {13, 5}, {416, 505}, {331, 162},
      {141, 136}, {84, 93},
      {254, 301}, {56, 44}, {12, 375}, {46, 61}, {43, 48}, {164, 184}, {179, 53}, {385, 473},
      {267, 307}, {33, 34},
      {163, 183}, {221, 283}, {223, 285}, {354, 433}, {107, 20}, {193, 211}, {136, 153}, {184, 200},
      {200, 222},
      {215, 252}, {204, 228}, {177, 58}, {319, 33}, {110, 116}, {116, 124}, {145, 165}, {99, 115},
      {360, 440},
      {407, 497}, {284, 335}, {121, 130}, {214, 240}, {373, 448}, {94, 103}, {150, 170}, {95, 107},
      {346, 247},
      {246, 290}, {336, 396}, {240, 280}, {87, 99}, {126, 141}, {191, 405}, {174, 194}, {106, 119},
      {159, 179},
      {348, 423}, {93, 102}, {342, 410}, {138, 157}, {289, 338}, {168, 394}, {36, 374}, {96, 110},
      {182, 197},
      {339, 406}, {255, 298}, {98, 111}, {257, 341}, {89, 104}, {119, 129}, {65, 78}, {269, 319},
      {288, 337},
      {227, 263}, {356, 438}, {90, 105}, {194, 212}, {205, 229}, {270, 320}, {132, 138}, {127, 142},
      {185, 401},
      {39, 45}, {285, 342}, {134, 144}, {404, 495}, {161, 181}, {186, 403}, {190, 204}, {369, 447},
      {306, 353},
      {100, 382}, {395, 484}, {129, 140}, {103, 120}, {394, 480}, {417, 506}, {50, 42}, {72, 82},
      {35, 43}, {213, 239},
      {69, 95}, {303, 351}, {368, 445}, {361, 441}, {337, 400}, {171, 52}, {247, 291}, {256, 303},
      {228, 264},
      {187, 207}, {335, 186}, {393, 481}, {178, 195}, {42, 47}, {92, 106}, {62, 75}, {156, 176},
      {137, 155}, {330, 383},
      {382, 391}, {353, 432}, {318, 11}, {243, 279}, {135, 143}, {198, 213}, {386, 299}, {147, 167},
      {160, 180},
      {268, 318}, {242, 274}, {68, 94}, {291, 352}, {264, 312}, {123, 135}, {341, 407}, {53, 66},
      {1, 2}, {387, 476},
      {409, 485}, {114, 123}, {109, 118}, {124, 131}, {222, 284}, {388, 302}, {73, 83}, {183, 198},
      {355, 474},
      {49, 64}, {151, 178}, {154, 171}, {71, 69}, {327, 378}, {149, 169}, {188, 208}, {181, 196},
      {211, 244},
      {277, 334}, {196, 217}, {167, 132}, {349, 425}, {278, 331}, {82, 97}, {201, 203}, {237, 268},
      {260, 306},
      {199, 219}, {340, 464}, {365, 443}, {271, 317}, {77, 86}, {231, 267}, {104, 122}, {263, 311},
      {265, 313},
      {64, 77}, {244, 278}, {61, 377}, {326, 121}, {383, 465}, {158, 177}, {253, 300}, {157, 175},
      {279, 333},
      {143, 172}, {113, 126}, {322, 369}, {74, 81}, {351, 430}, {389, 483}, {307, 354}, {232, 243},
      {313, 359},
      {281, 325}, {381, 459}, {343, 411}, {75, 80}, {26, 24}, {38, 46}, {296, 349}, {378, 381},
      {308, 330}, {414, 51},
      {274, 314}, {212, 494}, {91, 100}, {224, 286}, {266, 358}, {397, 488}, {176, 57}, {295, 350},
      {405, 496},
      {41, 39}, {148, 168}, {166, 189}, {367, 329}, {85, 89}, {292, 343}, {81, 379}, {333, 461},
      {239, 272}, {170, 133},
      {117, 125}, {59, 72}, {197, 220}, {248, 434}, {11, 19}, {10, 21}, {258, 309}, {51, 41},
      {86, 101}, {216, 251},
      {208, 235}, {131, 147}, {101, 117}, {251, 437}, {241, 416}, {18, 27}, {261, 439}, {236, 271},
      {338, 205},
      {359, 310}, {320, 26}, {286, 346}, {7, 29}, {40, 35}, {207, 233}, {217, 419}, {54, 30},
      {31, 36}, {406, 498},
      {398, 487}, {175, 50},
  };

//  std::cout << "matchResult: " << matchResult << std::endl;

  ASSERT_EQ(275, matchResult.size());
  for (int i = 0; i < matchResult.size(); i++) {
//    std::cout << "{" << matchResult[i].first << ", " << matchResult[i].second << "}," << std::endl;
    ASSERT_EQ(expectedMatch[i].first, matchResult[i].first);
    ASSERT_EQ(expectedMatch[i].second, matchResult[i].second);
  }
}

TEST(TLBD, line_matching_example_boat) {

  //load first image from file
  cv::Mat cvLeftImage = cv::imread("../resources/boat1.jpg", cv::IMREAD_GRAYSCALE);
  cv::Mat cvRightImage = cv::imread("../resources/boat3.jpg", cv::IMREAD_GRAYSCALE);


  // 2.1. Detecting lines in the scale space
  eth::MultiOctaveSegmentDetector detector(std::make_shared<eth::EDLineDetector>());
  eth::ScaleLines linesInLeft = detector.octaveKeyLines(cvLeftImage);
  eth::ScaleLines linesInRight = detector.octaveKeyLines(cvRightImage);

  // Description: 2.2. The band representation of the line support region & 2.3. The construction of the Line Band Descriptor
  eth::LineBandDescriptor lineDesc;
  std::vector<std::vector<cv::Mat>> descriptorsLeft, descriptorsRight;
  lineDesc.compute(cvLeftImage, linesInLeft, descriptorsLeft);
  lineDesc.compute(cvRightImage, linesInRight, descriptorsRight);

  // 3. Graph matching using spectral technique
  std::vector<std::pair<uint32_t, uint32_t>> matchResult;
  eth::PairwiseLineMatching lineMatch;
  lineMatch.matchLines(linesInLeft, linesInRight, descriptorsLeft, descriptorsRight, matchResult);

  // Show the result
//  draw_matches(cvLeftImage, cvRightImage, linesInLeft, linesInRight, matchResult);

  std::vector<std::pair<uint32_t, uint32_t>> expectedMatch = {
      {369, 118}, {236, 147}, {227, 144}, {335, 101}, {699, 395}, {349, 143}, {163, 63}, {58, 114},
      {641, 310}, {123, 132}, {137, 43}, {425, 232}, {293, 182}, {184, 91}, {533, 239}, {922, 467},
      {764, 220}, {4, 80}, {326, 141}, {112, 98}, {551, 267}, {452, 289}, {758, 178}, {707, 107},
      {757, 151}, {0, 57}, {692, 95}, {75, 94}, {879, 276}, {258, 212}, {903, 457}, {531, 277},
      {285, 162}, {64, 99}, {851, 485}, {466, 245}, {826, 240}, {277, 214}, {51, 60}, {419, 198},
      {376, 236}, {212, 157}, {747, 129}, {626, 241}, {712, 222}, {441, 286}, {508, 268},
      {789, 264}, {890, 433}, {731, 123}, {761, 235}, {325, 216}, {746, 218}, {732, 170}, {79, 93},
      {774, 199}, {196, 155}, {575, 260}, {929, 448}, {177, 175}, {218, 111}, {854, 70}, {152, 104},
      {59, 149}, {653, 473}, {811, 327}, {744, 213}, {418, 197}, {548, 292}, {878, 470}, {215, 125},
      {294, 181}, {781, 279}, {756, 161}, {360, 164}, {379, 414}, {316, 183}, {484, 265},
      {290, 184}, {127, 177}, {814, 281}, {156, 39}, {940, 455}, {673, 303}, {738, 412}, {69, 116},
      {197, 201}, {725, 131}, {846, 134}, {34, 21}, {305, 180}, {595, 417}, {57, 113}, {553, 284},
      {298, 255}, {191, 392}, {804, 259}, {925, 422}, {919, 269}, {367, 409}, {403, 211},
      {911, 464}, {877, 280}, {745, 185}, {813, 318}, {713, 223}, {407, 221}, {554, 261}, {22, 74},
      {905, 483}, {465, 204}, {329, 126}, {174, 174}, {801, 306}, {9, 58}, {320, 193}, {331, 225},
      {564, 253}, {938, 487}, {283, 192}, {924, 466}, {337, 224}, {280, 458}, {308, 228}, {60, 121},
      {556, 305}, {327, 160}, {881, 461}, {109, 122}, {872, 431}, {44, 62}, {547, 304}, {780, 282},
      {809, 301}, {446, 249}, {141, 44}, {241, 130}, {460, 416}, {302, 153}, {275, 187}, {317, 169},
      {68, 83}, {87, 135}, {153, 92}, {734, 454}, {318, 195}, {89, 137}, {496, 246}, {824, 312},
      {948, 490}, {769, 258}, {71, 96}, {378, 205}, {912, 179}, {29, 82}, {786, 202}, {442, 426},
      {656, 298}, {205, 450}, {400, 152}, {512, 244}, {529, 293}, {767, 315}, {684, 49}, {552, 285},
      {11, 12}, {28, 46}, {35, 391}, {231, 154}, {27, 85}, {580, 326}, {313, 115}, {252, 120},
      {111, 150}, {946, 489}, {812, 317}, {385, 252}, {776, 425}, {248, 190}, {348, 102},
      {768, 270}, {451, 295}, {651, 233}, {138, 108}, {493, 203}, {779, 291}, {364, 89}, {520, 247},
      {603, 320}, {832, 272}, {709, 140}, {2, 71}, {192, 90}, {701, 397}, {749, 163}, {370, 119},
      {63, 100}, {798, 307}, {198, 69}, {787, 158}, {288, 243}, {264, 139}, {203, 146}, {818, 328},
      {805, 283}, {821, 313}, {729, 165}, {350, 226}, {121, 208}, {823, 430}, {820, 334},
      {649, 273}, {689, 447}, {304, 128}, {735, 127}, {573, 321}, {759, 117}, {300, 256}, {172, 40},
      {644, 230}, {538, 302}, {504, 262}, {676, 325}, {698, 97}, {159, 79}, {532, 278}, {254, 234},
      {447, 254}, {655, 238}, {632, 299}, {642, 421}, {594, 266}, {15, 7}, {737, 173}, {481, 300},
      {840, 482},
  };

//  for (int i = 0; i < matchResult.size(); i++) {
//    std::cout << "{" << matchResult[i].first << ", " << matchResult[i].second << "}," << std::endl;
//  }

  ASSERT_EQ(236, matchResult.size());
  for (int i = 0; i < matchResult.size(); i++) {
//    std::cout << "{" << matchResult[i].first << ", " << matchResult[i].second << "}," << std::endl;
    ASSERT_EQ(expectedMatch[i].first, matchResult[i].first);
    ASSERT_EQ(expectedMatch[i].second, matchResult[i].second);
  }
}

TEST(TLBD, line_matching_example_graf) {

  //load first image from file
  cv::Mat cvLeftImage = cv::imread("../resources/graf1.ppm", cv::IMREAD_GRAYSCALE);
  cv::Mat cvRightImage = cv::imread("../resources/graf2.ppm", cv::IMREAD_GRAYSCALE);


  // 2.1. Detecting lines in the scale space
  eth::MultiOctaveSegmentDetector detector(std::make_shared<eth::EDLineDetector>());
  eth::ScaleLines linesInLeft = detector.octaveKeyLines(cvLeftImage);
  eth::ScaleLines linesInRight = detector.octaveKeyLines(cvRightImage);

  // Description: 2.2. The band representation of the line support region & 2.3. The construction of the Line Band Descriptor
  eth::LineBandDescriptor lineDesc;
  std::vector<std::vector<cv::Mat>> descriptorsLeft, descriptorsRight;
  lineDesc.compute(cvLeftImage, linesInLeft, descriptorsLeft);
  lineDesc.compute(cvRightImage, linesInRight, descriptorsRight);

  // 3. Graph matching using spectral technique
  std::vector<std::pair<uint32_t, uint32_t>> matchResult;
  eth::PairwiseLineMatching lineMatch;
  lineMatch.matchLines(linesInLeft, linesInRight, descriptorsLeft, descriptorsRight, matchResult);

  // Show the result
//  draw_matches(cvLeftImage, cvRightImage, linesInLeft, linesInRight, matchResult);

  std::vector<std::pair<uint32_t, uint32_t>> expectedMatch = {
      {150, 184}, {103, 228}, {939, 1006}, {105, 189}, {89, 168}, {174, 295}, {391, 473}, {1025, 1064}, {172, 457},
      {558, 554}, {210, 232}, {1041, 1072}, {167, 445}, {545, 550}, {307, 404}, {90, 169}, {395, 347}, {298, 812},
      {215, 264}, {798, 401}, {132, 970}, {226, 233}, {345, 413}, {995, 904}, {256, 258}, {767, 821}, {886, 616},
      {794, 997}, {1072, 1098}, {762, 771}, {164, 263}, {402, 380}, {930, 976}, {769, 851}, {121, 262}, {960, 1008},
      {168, 444}, {159, 224}, {772, 798}, {519, 450}, {777, 234}, {1075, 1105}, {93, 176}, {151, 185}, {799, 795},
      {479, 429}, {177, 259}, {99, 159}, {898, 583}, {937, 990}, {92, 175}, {1079, 1075}, {47, 757}, {525, 480},
      {145, 187}, {824, 846}, {67, 106}, {821, 283}, {773, 210}, {1051, 1029}, {992, 545}, {378, 432}, {346, 412},
      {786, 388}, {1057, 998}, {936, 1000}, {795, 358}, {1034, 1004}, {407, 378}, {46, 78}, {74, 91}, {30, 179},
      {1024, 1062}, {173, 294}, {481, 430}, {254, 344}, {986, 1032}, {990, 909}, {63, 85}, {414, 497}, {347, 410},
      {147, 138}, {952, 992}, {116, 134}, {263, 287}, {186, 787}, {547, 555}, {744, 118}, {207, 162}, {134, 137},
      {358, 818}, {97, 206}, {768, 820}, {892, 607}, {1035, 1085}, {311, 291}, {853, 435}, {583, 547}, {877, 594},
      {114, 79}, {343, 407}, {931, 966}, {771, 855}, {217, 289}, {832, 517}, {377, 397}, {257, 257}, {344, 408},
      {11, 18}, {262, 267}, {1062, 1036}, {816, 848}, {743, 117}, {249, 299}, {125, 156}, {430, 436}, {781, 804},
      {71, 110}, {124, 157}, {369, 446}, {88, 170}, {514, 461}, {335, 350}, {183, 220}, {28, 116}, {231, 277},
      {137, 130}, {367, 1007}, {238, 183}, {315, 309}, {431, 437}, {835, 1001}, {220, 366}, {245, 813}, {823, 398},
      {755, 133}, {1032, 1070}, {104, 174}, {241, 316}, {372, 409}, {95, 83}, {498, 514}, {130, 191}, {324, 312},
      {908, 574}, {68, 107}, {812, 531}, {925, 128}, {933, 968}, {323, 311}, {227, 296}, {934, 996}, {236, 363},
      {193, 314}, {312, 290}, {845, 887}, {753, 776}, {815, 419}, {415, 496}, {189, 207}, {212, 333}, {259, 255},
      {879, 541}, {833, 307}, {993, 1030}, {180, 337}, {362, 411}, {258, 256}, {199, 254}, {149, 34}, {38, 69},
      {462, 542}, {200, 253}, {62, 127}, {119, 167}, {213, 203}, {534, 598}, {184, 221}, {604, 540}, {513, 587},
      {341, 373}, {350, 393}, {100, 160}, {53, 146}, {39, 68}, {337, 361}, {775, 794}, {465, 586}, {912, 618},
      {825, 844}, {127, 273}, {409, 527}, {250, 300}, {269, 332}, {182, 219}, {148, 33}, {455, 468}, {410, 484},
      {491, 508}, {542, 502}, {426, 449}, {52, 147}, {813, 495}, {383, 516}, {219, 365}, {123, 151}, {208, 230},
      {237, 362}, {854, 529}, {468, 503}, {1026, 1100}, {6, 41}, {131, 192}, {64, 82}, {915, 600}, {463, 544},
      {49, 142}, {223, 370}, {620, 564}, {283, 367}, {55, 140}, {235, 353}, {106, 190}, {809, 447}, {353, 355},
      {890, 619}, {270, 387}, {185, 304}, {452, 591}, {738, 104}, {373, 382}, {876, 499}, {394, 346}, {784, 815},
      {187, 209}, {354, 357}, {457, 425}, {325, 313}, {234, 354}, {146, 188}, {543, 585}, {14, 70}, {205, 335},
      {829, 621}, {366, 390}, {740, 1053}, {582, 570}, {976, 863}, {406, 377}, {396, 348}, {826, 839}, {523, 599},
      {849, 889}, {190, 226}, {654, 629}, {306, 356}, {120, 166}, {839, 376}, {655, 630}, {663, 525}, {429, 438},
      {300, 270}, {852, 856}, {143, 286}, {433, 532}, {118, 132}, {766, 782}, {549, 493}, {471, 477}, {188, 792},
      {260, 325}, {85, 14}, {83, 12}, {292, 305}, {386, 518}, {913, 928}, {978, 584}, {828, 403}, {848, 421},
      {790, 280}, {338, 385}, {742, 1003}, {959, 837}, {94, 177}, {98, 165}, {393, 426}, {488, 482}, {501, 563},
      {470, 478}, {54, 141}, {539, 500}, {916, 602}, {368, 505}, {602, 533}, {464, 592}, {287, 340}, {546, 551},
      {218, 364}, {424, 448}, {61, 126}, {878, 536}, {411, 504}, {472, 476}, {572, 512}, {856, 865}, {504, 528},
      {359, 817}, {252, 343}, {778, 161}, {989, 908}, {756, 761}, {122, 150}, {973, 849}, {893, 609}, {0, 9},
      {267, 392}, {232, 276}, {221, 198}, {264, 288}, {401, 507}, {637, 539}, {202, 252}, {305, 349}, {214, 163},
      {216, 265}, {389, 439}, {797, 395}, {246, 301}, {863, 871}, {838, 1073}, {230, 215}, {87, 16}, {633, 612},
      {567, 452}, {244, 303}, {900, 597}, {163, 199}, {605, 622}, {941, 319}, {782, 223}, {512, 417}, {1033, 974},
      {961, 825}, {203, 251}, {817, 802}, {322, 310}, {540, 501}, {975, 864}, {349, 341}, {191, 322}, {1008, 883},
      {691, 576}, {229, 331}, {16, 97}, {814, 999}, {855, 451}, {612, 866}, {803, 810}, {485, 857}, {444, 560},
      {895, 538}, {13, 20}, {718, 603}, {70, 109}, {403, 374}, {166, 148}, {48, 66}, {1013, 905}, {791, 819}, {10, 17},
      {475, 467}, {831, 553}, {117, 962}, {428, 384}, {550, 492}, {950, 814}, {1007, 888}, {692, 577}, {408, 379},
      {596, 854}, {844, 886}, {720, 601}, {376, 396}, {496, 513}, {1064, 1033}, {388, 282}, {764, 122}, {632, 611},
      {704, 648}, {730, 660}, {1002, 882}, {72, 1056}, {628, 608}, {333, 321}, {286, 285}, {357, 327}, {575, 1010},
      {728, 522}, {996, 624}, {865, 475}, {804, 323}, {9, 149}, {509, 511}, {502, 466}, {319, 318}
  };

//  for (int i = 0; i < matchResult.size(); i++) {
//    std::cout << "{" << matchResult[i].first << ", " << matchResult[i].second << "}," << std::endl;
//  }

  ASSERT_EQ(405, matchResult.size());
  for (int i = 0; i < matchResult.size(); i++) {
//    std::cout << "{" << matchResult[i].first << ", " << matchResult[i].second << "}," << std::endl;
    ASSERT_EQ(expectedMatch[i].first, matchResult[i].first);
    ASSERT_EQ(expectedMatch[i].second, matchResult[i].second);
  }
}

TEST(MultiScaleSegments, DuplicatedSegments) {
  cv::Mat img = cv::imread("../resources/boat1.jpg", cv::IMREAD_GRAYSCALE);

  EDLineDetector edlines;
  SalientSegments segs = edlines.detectSalient(img);

  ScaleLines expected;
  for (SalientSegment s : segs) {
    expected.push_back({keyline_from_seg(s.segment), keyline_from_seg(s.segment)});
  }

  // Checks that the method is able to group perfectly aligned segments
  std::vector<Segments> octaveSegs(2);
  std::vector<std::vector<float>> saliencies(2);
  std::vector<std::vector<size_t>> nPixels(2);
  for (SalientSegment &ss : segs) {
    octaveSegs[0].push_back(ss.segment);
    octaveSegs[1].push_back(ss.segment / M_SQRT2);
    saliencies[0].push_back(ss.salience);
    saliencies[1].push_back(ss.salience);
    nPixels[0].push_back((size_t) math::segLength(ss.segment));
    nPixels[1].push_back(size_t(math::segLength(ss.segment) / M_SQRT2));
  }

  ScaleLines mergedSegs = MultiOctaveSegmentDetector::mergeOctaveLines(octaveSegs, saliencies, nPixels);

  ASSERT_EQ(segs.size(), mergedSegs.size());
  for (auto &lineVec : mergedSegs) {
    ASSERT_EQ(2, lineVec.size());
    ASSERT_FLOAT_EQ(lineVec[0].startPointX, lineVec[1].startPointX);
    ASSERT_FLOAT_EQ(lineVec[0].startPointY, lineVec[1].startPointY);
    ASSERT_FLOAT_EQ(lineVec[0].endPointX, lineVec[1].endPointX);
    ASSERT_FLOAT_EQ(lineVec[0].endPointY, lineVec[1].endPointY);
  }
}

int main(int argc, char **argv) {
  std::cout << "**********************************************" << std::endl;
  std::cout << "****************** TLBD Test *****************" << std::endl;
  std::cout << "**********************************************" << std::endl;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}