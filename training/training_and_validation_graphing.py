#!/usr/bin/env python

import matplotlib.pyplot as plt

# Sample data from the provided text
data = """epoch: 1, step: 3, time: 37.1, train: 1.043064, valid: 0.007375, train_acc: 1.004960, valid_acc: 0.004381
epoch: 2, step: 6, time: 36.4, train: 1.041946, valid: 0.007600, train_acc: 1.003886, valid_acc: 0.004779
epoch: 3, step: 9, time: 39.9, train: 1.079868, valid: 0.011674, train_acc: 1.004752, valid_acc: 0.005974
epoch: 4, step: 12, time: 33.9, train: 1.078347, valid: 0.011591, train_acc: 1.002635, valid_acc: 0.006173
epoch: 5, step: 15, time: 33.4, train: 1.070374, valid: 0.015848, train_acc: 0.997795, valid_acc: 0.005575
epoch: 6, step: 18, time: 29.7, train: 1.069268, valid: 0.014553, train_acc: 0.995441, valid_acc: 0.005974
epoch: 7, step: 21, time: 40.9, train: 1.065216, valid: 0.010969, train_acc: 0.977739, valid_acc: 0.004630
epoch: 8, step: 24, time: 36.9, train: 1.062182, valid: 0.010767, train_acc: 0.973436, valid_acc: 0.004480
epoch: 9, step: 27, time: 44.0, train: 1.057000, valid: 0.011208, train_acc: 0.968494, valid_acc: 0.004779
epoch: 10, step: 30, time: 40.9, train: 1.051738, valid: 0.011137, train_acc: 0.963289, valid_acc: 0.005227
epoch: 11, step: 33, time: 34.6, train: 1.053019, valid: 0.017056, train_acc: 0.969709, valid_acc: 0.007368
epoch: 12, step: 36, time: 31.5, train: 1.044945, valid: 0.018477, train_acc: 0.963274, valid_acc: 0.007567
epoch: 13, step: 38, time: 30.6, train: 1.022214, valid: 0.017924, train_acc: 0.938462, valid_acc: 0.007019
epoch: 14, step: 40, time: 29.3, train: 1.016868, valid: 0.019107, train_acc: 0.933897, valid_acc: 0.008363
epoch: 15, step: 43, time: 45.1, train: 1.033281, valid: 0.021209, train_acc: 0.934676, valid_acc: 0.008961
epoch: 16, step: 46, time: 37.6, train: 1.023096, valid: 0.023830, train_acc: 0.926974, valid_acc: 0.010603
epoch: 17, step: 49, time: 36.3, train: 1.005475, valid: 0.024032, train_acc: 0.931130, valid_acc: 0.015333
epoch: 18, step: 52, time: 34.5, train: 0.994103, valid: 0.029732, train_acc: 0.922111, valid_acc: 0.017125
epoch: 19, step: 54, time: 39.7, train: 0.942908, valid: 0.032373, train_acc: 0.914470, valid_acc: 0.017324
epoch: 20, step: 56, time: 34.0, train: 0.936293, valid: 0.034616, train_acc: 0.907885, valid_acc: 0.018519
epoch: 21, step: 58, time: 32.6, train: 0.948167, valid: 0.029390, train_acc: 0.889505, valid_acc: 0.014934
epoch: 22, step: 60, time: 28.0, train: 0.941526, valid: 0.030138, train_acc: 0.882885, valid_acc: 0.016428
epoch: 23, step: 63, time: 42.5, train: 0.957155, valid: 0.046431, train_acc: 0.884372, valid_acc: 0.022501
epoch: 24, step: 66, time: 38.7, train: 0.947064, valid: 0.050106, train_acc: 0.873562, valid_acc: 0.025289
epoch: 25, step: 69, time: 46.0, train: 0.930762, valid: 0.054888, train_acc: 0.852886, valid_acc: 0.023447
epoch: 26, step: 72, time: 38.0, train: 0.919830, valid: 0.059260, train_acc: 0.841396, valid_acc: 0.027479
epoch: 27, step: 74, time: 34.8, train: 0.870325, valid: 0.062969, train_acc: 0.832426, valid_acc: 0.029122
epoch: 28, step: 76, time: 30.2, train: 0.861612, valid: 0.068785, train_acc: 0.824200, valid_acc: 0.033453
epoch: 29, step: 79, time: 40.4, train: 0.879191, valid: 0.070555, train_acc: 0.823588, valid_acc: 0.047790
epoch: 30, step: 82, time: 33.8, train: 0.866482, valid: 0.261066, train_acc: 0.810695, valid_acc: 0.595579
epoch: 31, step: 85, time: 39.3, train: 0.861784, valid: 0.432938, train_acc: 0.798304, valid_acc: 0.600358
epoch: 32, step: 88, time: 34.5, train: 0.848985, valid: 0.442374, train_acc: 0.785311, valid_acc: 0.608722
epoch: 33, step: 91, time: 40.1, train: 0.825556, valid: 0.486709, train_acc: 0.769832, valid_acc: 0.626643
epoch: 34, step: 94, time: 33.8, train: 0.810035, valid: 0.498739, train_acc: 0.756042, valid_acc: 0.637196
epoch: 35, step: 97, time: 41.2, train: 0.813935, valid: 0.456005, train_acc: 0.733647, valid_acc: 0.673984
epoch: 36, step: 100, time: 36.3, train: 0.799702, valid: 0.473627, train_acc: 0.719469, valid_acc: 0.687575
epoch: 37, step: 102, time: 27.1, train: 0.757364, valid: 0.584457, train_acc: 0.702980, valid_acc: 0.718339
epoch: 38, step: 104, time: 25.1, train: 0.748212, valid: 0.593403, train_acc: 0.693869, valid_acc: 0.724761
epoch: 39, step: 107, time: 38.0, train: 0.740095, valid: 0.597231, train_acc: 0.695259, valid_acc: 0.690761
epoch: 40, step: 110, time: 34.7, train: 0.724464, valid: 0.617379, train_acc: 0.680604, valid_acc: 0.710872
epoch: 41, step: 112, time: 27.9, train: 0.696110, valid: 0.687190, train_acc: 0.670135, valid_acc: 0.731581
epoch: 42, step: 114, time: 22.9, train: 0.685330, valid: 0.706805, train_acc: 0.660282, valid_acc: 0.745321
epoch: 43, step: 117, time: 36.3, train: 0.698383, valid: 0.683515, train_acc: 0.647568, valid_acc: 0.755874
epoch: 44, step: 120, time: 37.3, train: 0.682485, valid: 0.710232, train_acc: 0.632636, valid_acc: 0.781362
epoch: 45, step: 123, time: 36.3, train: 0.682417, valid: 0.692770, train_acc: 0.616140, valid_acc: 0.821187
epoch: 46, step: 126, time: 31.8, train: 0.665360, valid: 0.730457, train_acc: 0.600965, valid_acc: 0.842294
epoch: 47, step: 128, time: 35.8, train: 0.612114, valid: 0.817079, train_acc: 0.590936, valid_acc: 0.856033
epoch: 48, step: 130, time: 29.5, train: 0.601584, valid: 0.831354, train_acc: 0.581055, valid_acc: 0.866786
epoch: 49, step: 134, time: 49.1, train: 0.619745, valid: 0.791995, train_acc: 0.561362, valid_acc: 0.900438
epoch: 50, step: 138, time: 42.8, train: 0.596799, valid: 0.836865, train_acc: 0.541458, valid_acc: 0.923736
epoch: 51, step: 141, time: 42.2, train: 0.566359, valid: 0.871938, train_acc: 0.526786, valid_acc: 0.935882
epoch: 52, step: 144, time: 36.6, train: 0.550848, valid: 0.894924, train_acc: 0.512182, valid_acc: 0.947431
epoch: 53, step: 147, time: 44.5, train: 0.534052, valid: 0.916076, train_acc: 0.498400, valid_acc: 0.954202
epoch: 54, step: 150, time: 42.7, train: 0.519306, valid: 0.932374, train_acc: 0.483796, valid_acc: 0.965751
epoch: 55, step: 152, time: 28.5, train: 0.506611, valid: 0.943863, train_acc: 0.473704, valid_acc: 0.971724
epoch: 56, step: 154, time: 25.2, train: 0.496393, valid: 0.953246, train_acc: 0.464281, valid_acc: 0.975309
epoch: 57, step: 157, time: 45.2, train: 0.493146, valid: 0.956023, train_acc: 0.450390, valid_acc: 0.976902
epoch: 58, step: 160, time: 40.7, train: 0.479312, valid: 0.965176, train_acc: 0.436756, valid_acc: 0.981880
epoch: 59, step: 163, time: 34.5, train: 0.460456, valid: 0.979226, train_acc: 0.415564, valid_acc: 0.987754
epoch: 60, step: 166, time: 28.3, train: 0.443730, valid: 0.985923, train_acc: 0.402602, valid_acc: 0.991637
epoch: 61, step: 169, time: 35.2, train: 0.431861, valid: 0.989684, train_acc: 0.396170, valid_acc: 0.994225
epoch: 62, step: 172, time: 31.2, train: 0.417560, valid: 0.993073, train_acc: 0.383188, valid_acc: 0.996018
epoch: 63, step: 175, time: 43.9, train: 0.397336, valid: 0.993749, train_acc: 0.369936, valid_acc: 0.997611
epoch: 64, step: 178, time: 39.0, train: 0.384015, valid: 0.995857, train_acc: 0.357542, valid_acc: 0.998407
epoch: 65, step: 181, time: 43.3, train: 0.371374, valid: 0.997649, train_acc: 0.346698, valid_acc: 0.997013
epoch: 66, step: 184, time: 38.2, train: 0.359124, valid: 0.998218, train_acc: 0.334777, valid_acc: 0.997212
epoch: 67, step: 186, time: 26.6, train: 0.344647, valid: 0.998487, train_acc: 0.327104, valid_acc: 0.997212
epoch: 68, step: 188, time: 22.5, train: 0.337087, valid: 0.998971, train_acc: 0.319459, valid_acc: 0.997411
epoch: 69, step: 191, time: 40.5, train: 0.333548, valid: 0.999205, train_acc: 0.306323, valid_acc: 0.999403
epoch: 70, step: 194, time: 33.8, train: 0.321269, valid: 0.999426, train_acc: 0.295582, valid_acc: 0.999851
epoch: 71, step: 197, time: 36.0, train: 0.314869, valid: 0.999638, train_acc: 0.286259, valid_acc: 1.000000
epoch: 72, step: 200, time: 29.3, train: 0.302053, valid: 0.999897, train_acc: 0.275875, valid_acc: 1.000000
epoch: 73, step: 203, time: 38.1, train: 0.287712, valid: 0.999911, train_acc: 0.265440, valid_acc: 1.000000
epoch: 74, step: 206, time: 35.0, train: 0.276130, valid: 1.000000, train_acc: 0.255567, valid_acc: 1.000000
epoch: 75, step: 209, time: 43.8, train: 0.267147, valid: 1.000000, train_acc: 0.243804, valid_acc: 1.000000
epoch: 76, step: 212, time: 36.5, train: 0.256724, valid: 1.000000, train_acc: 0.234815, valid_acc: 1.000000
epoch: 77, step: 215, time: 45.1, train: 0.247072, valid: 1.000000, train_acc: 0.228671, valid_acc: 1.000000
epoch: 78, step: 218, time: 36.8, train: 0.237925, valid: 1.000000, train_acc: 0.219925, valid_acc: 1.000000
epoch: 79, step: 221, time: 37.1, train: 0.225111, valid: 1.000000, train_acc: 0.209355, valid_acc: 1.000000
epoch: 80, step: 224, time: 30.0, train: 0.217236, valid: 1.000000, train_acc: 0.201446, valid_acc: 1.000000
epoch: 81, step: 227, time: 48.5, train: 0.207502, valid: 1.000000, train_acc: 0.193941, valid_acc: 1.000000
epoch: 82, step: 230, time: 45.1, train: 0.199507, valid: 1.000000, train_acc: 0.186683, valid_acc: 1.000000
epoch: 83, step: 233, time: 43.6, train: 0.193938, valid: 1.000000, train_acc: 0.179755, valid_acc: 1.000000
epoch: 84, step: 236, time: 30.8, train: 0.186694, valid: 1.000000, train_acc: 0.172994, valid_acc: 1.000000
epoch: 85, step: 239, time: 41.0, train: 0.180503, valid: 1.000000, train_acc: 0.167292, valid_acc: 1.000000
epoch: 86, step: 242, time: 37.7, train: 0.173752, valid: 1.000000, train_acc: 0.160820, valid_acc: 1.000000
epoch: 87, step: 245, time: 36.4, train: 0.163885, valid: 1.000000, train_acc: 0.154881, valid_acc: 1.000000
epoch: 88, step: 248, time: 29.1, train: 0.157935, valid: 1.000000, train_acc: 0.148925, valid_acc: 1.000000
epoch: 89, step: 250, time: 33.9, train: 0.148552, valid: 1.000000, train_acc: 0.144681, valid_acc: 1.000000
epoch: 90, step: 252, time: 32.2, train: 0.144763, valid: 1.000000, train_acc: 0.141040, valid_acc: 1.000000
epoch: 91, step: 255, time: 41.5, train: 0.145992, valid: 1.000000, train_acc: 0.135649, valid_acc: 1.000000
epoch: 92, step: 258, time: 36.3, train: 0.139828, valid: 1.000000, train_acc: 0.130388, valid_acc: 1.000000
epoch: 93, step: 260, time: 28.1, train: 0.132153, valid: 1.000000, train_acc: 0.126999, valid_acc: 1.000000
epoch: 94, step: 262, time: 25.8, train: 0.128856, valid: 1.000000, train_acc: 0.123640, valid_acc: 1.000000
epoch: 95, step: 265, time: 38.4, train: 0.125059, valid: 1.000000, train_acc: 0.118893, valid_acc: 1.000000
epoch: 96, step: 268, time: 34.1, train: 0.119788, valid: 1.000000, train_acc: 0.114275, valid_acc: 1.000000
epoch: 97, step: 271, time: 39.2, train: 0.116093, valid: 1.000000, train_acc: 0.109855, valid_acc: 1.000000
epoch: 98, step: 274, time: 38.9, train: 0.111405, valid: 1.000000, train_acc: 0.105609, valid_acc: 1.000000
epoch: 99, step: 277, time: 43.3, train: 0.106163, valid: 1.000000, train_acc: 0.101529, valid_acc: 1.000000
epoch: 100, step: 280, time: 37.8, train: 0.101876, valid: 1.000000, train_acc: 0.097518, valid_acc: 1.000000
epoch: 101, step: 283, time: 38.5, train: 0.099217, valid: 1.000000, train_acc: 0.093902, valid_acc: 1.000000
epoch: 102, step: 286, time: 35.6, train: 0.095489, valid: 1.000000, train_acc: 0.090233, valid_acc: 1.000000
epoch: 103, step: 288, time: 32.7, train: 0.090190, valid: 1.000000, train_acc: 0.088179, valid_acc: 1.000000
epoch: 104, step: 290, time: 27.7, train: 0.087767, valid: 1.000000, train_acc: 0.085932, valid_acc: 1.000000
epoch: 105, step: 293, time: 42.4, train: 0.086195, valid: 1.000000, train_acc: 0.082684, valid_acc: 1.000000
epoch: 106, step: 296, time: 36.7, train: 0.082796, valid: 1.000000, train_acc: 0.079598, valid_acc: 1.000000
epoch: 107, step: 300, time: 43.5, train: 0.080202, valid: 1.000000, train_acc: 0.075104, valid_acc: 1.000000
epoch: 108, step: 304, time: 36.5, train: 0.076002, valid: 1.000000, train_acc: 0.071235, valid_acc: 1.000000
epoch: 109, step: 306, time: 30.6, train: 0.071537, valid: 1.000000, train_acc: 0.069399, valid_acc: 1.000000
epoch: 110, step: 308, time: 26.5, train: 0.069719, valid: 1.000000, train_acc: 0.067650, valid_acc: 1.000000
epoch: 111, step: 311, time: 40.8, train: 0.068084, valid: 1.000000, train_acc: 0.064972, valid_acc: 1.000000
epoch: 112, step: 314, time: 35.3, train: 0.065311, valid: 1.000000, train_acc: 0.062480, valid_acc: 1.000000
epoch: 113, step: 317, time: 36.5, train: 0.061498, valid: 1.000000, train_acc: 0.060246, valid_acc: 1.000000
epoch: 114, step: 320, time: 31.6, train: 0.059326, valid: 1.000000, train_acc: 0.057912, valid_acc: 1.000000
epoch: 115, step: 323, time: 33.9, train: 0.056718, valid: 1.000000, train_acc: 0.055696, valid_acc: 1.000000
epoch: 116, step: 326, time: 29.3, train: 0.054753, valid: 1.000000, train_acc: 0.053592, valid_acc: 1.000000
epoch: 117, step: 328, time: 24.2, train: 0.052274, valid: 1.000000, train_acc: 0.052230, valid_acc: 1.000000
epoch: 118, step: 330, time: 22.8, train: 0.050951, valid: 1.000000, train_acc: 0.050912, valid_acc: 1.000000
epoch: 119, step: 332, time: 27.4, train: 0.050284, valid: 1.000000, train_acc: 0.049672, valid_acc: 1.000000
epoch: 120, step: 334, time: 22.8, train: 0.049075, valid: 1.000000, train_acc: 0.048433, valid_acc: 1.000000
epoch: 121, step: 336, time: 28.8, train: 0.048246, valid: 1.000000, train_acc: 0.047182, valid_acc: 1.000000
epoch: 122, step: 338, time: 25.8, train: 0.046962, valid: 1.000000, train_acc: 0.046012, valid_acc: 1.000000
epoch: 123, step: 341, time: 40.6, train: 0.046089, valid: 1.000000, train_acc: 0.044448, valid_acc: 1.000000
epoch: 124, step: 344, time: 36.3, train: 0.044459, valid: 1.000000, train_acc: 0.042787, valid_acc: 1.000000
epoch: 125, step: 347, time: 43.1, train: 0.043297, valid: 1.000000, train_acc: 0.041888, valid_acc: 1.000000
epoch: 126, step: 350, time: 36.6, train: 0.041709, valid: 1.000000, train_acc: 0.040347, valid_acc: 1.000000
epoch: 127, step: 353, time: 36.7, train: 0.038881, valid: 1.000000, train_acc: 0.038859, valid_acc: 1.000000
epoch: 128, step: 356, time: 32.7, train: 0.037433, valid: 1.000000, train_acc: 0.037451, valid_acc: 1.000000
epoch: 129, step: 359, time: 41.6, train: 0.035991, valid: 1.000000, train_acc: 0.035426, valid_acc: 1.000000
epoch: 130, step: 362, time: 35.7, train: 0.034644, valid: 1.000000, train_acc: 0.034116, valid_acc: 1.000000
epoch: 131, step: 365, time: 48.8, train: 0.032431, valid: 1.000000, train_acc: 0.033930, valid_acc: 1.000000
epoch: 132, step: 368, time: 40.9, train: 0.031017, valid: 1.000000, train_acc: 0.032722, valid_acc: 1.000000
epoch: 133, step: 371, time: 38.5, train: 0.030754, valid: 1.000000, train_acc: 0.031156, valid_acc: 1.000000
epoch: 134, step: 374, time: 33.1, train: 0.029619, valid: 1.000000, train_acc: 0.030041, valid_acc: 1.000000
epoch: 135, step: 377, time: 42.8, train: 0.029097, valid: 1.000000, train_acc: 0.028375, valid_acc: 1.000000
epoch: 136, step: 380, time: 35.8, train: 0.028049, valid: 1.000000, train_acc: 0.027342, valid_acc: 1.000000
epoch: 137, step: 383, time: 36.6, train: 0.026031, valid: 1.000000, train_acc: 0.026955, valid_acc: 1.000000
epoch: 138, step: 386, time: 33.5, train: 0.025065, valid: 1.000000, train_acc: 0.025997, valid_acc: 1.000000
epoch: 139, step: 389, time: 34.9, train: 0.023808, valid: 1.000000, train_acc: 0.025051, valid_acc: 1.000000
epoch: 140, step: 392, time: 29.7, train: 0.022904, valid: 1.000000, train_acc: 0.024169, valid_acc: 1.000000
epoch: 141, step: 395, time: 35.0, train: 0.022734, valid: 1.000000, train_acc: 0.022769, valid_acc: 1.000000
epoch: 142, step: 398, time: 31.7, train: 0.021945, valid: 1.000000, train_acc: 0.021983, valid_acc: 1.000000
epoch: 143, step: 401, time: 41.0, train: 0.021682, valid: 1.000000, train_acc: 0.021198, valid_acc: 1.000000
epoch: 144, step: 404, time: 34.4, train: 0.020862, valid: 1.000000, train_acc: 0.020458, valid_acc: 1.000000
epoch: 145, step: 407, time: 37.7, train: 0.020065, valid: 1.000000, train_acc: 0.019702, valid_acc: 1.000000
epoch: 146, step: 410, time: 33.9, train: 0.019350, valid: 1.000000, train_acc: 0.018998, valid_acc: 1.000000
epoch: 147, step: 412, time: 25.0, train: 0.018085, valid: 1.000000, train_acc: 0.018550, valid_acc: 1.000000
epoch: 148, step: 414, time: 24.1, train: 0.017686, valid: 1.000000, train_acc: 0.018101, valid_acc: 1.000000
epoch: 149, step: 416, time: 26.5, train: 0.017122, valid: 1.000000, train_acc: 0.017675, valid_acc: 1.000000
epoch: 150, step: 418, time: 24.4, train: 0.016700, valid: 1.000000, train_acc: 0.017259, valid_acc: 1.000000
epoch: 151, step: 421, time: 35.1, train: 0.016360, valid: 1.000000, train_acc: 0.016683, valid_acc: 1.000000
epoch: 152, step: 424, time: 30.2, train: 0.015777, valid: 1.000000, train_acc: 0.016094, valid_acc: 1.000000
epoch: 153, step: 427, time: 43.4, train: 0.015973, valid: 1.000000, train_acc: 0.015981, valid_acc: 1.000000
epoch: 154, step: 430, time: 38.9, train: 0.015413, valid: 1.000000, train_acc: 0.015421, valid_acc: 1.000000
epoch: 155, step: 433, time: 35.3, train: 0.013787, valid: 1.000000, train_acc: 0.014458, valid_acc: 1.000000
epoch: 156, step: 436, time: 31.3, train: 0.013283, valid: 1.000000, train_acc: 0.013956, valid_acc: 1.000000
epoch: 157, step: 439, time: 39.8, train: 0.013152, valid: 1.000000, train_acc: 0.013481, valid_acc: 1.000000
epoch: 158, step: 442, time: 35.1, train: 0.012644, valid: 1.000000, train_acc: 0.013023, valid_acc: 1.000000
epoch: 159, step: 445, time: 42.2, train: 0.012436, valid: 1.000000, train_acc: 0.013007, valid_acc: 1.000000
epoch: 160, step: 448, time: 37.4, train: 0.012009, valid: 1.000000, train_acc: 0.012558, valid_acc: 1.000000
epoch: 161, step: 451, time: 42.4, train: 0.011523, valid: 1.000000, train_acc: 0.011719, valid_acc: 1.000000
epoch: 162, step: 454, time: 38.6, train: 0.011124, valid: 1.000000, train_acc: 0.011310, valid_acc: 1.000000
epoch: 163, step: 457, time: 33.9, train: 0.010637, valid: 1.000000, train_acc: 0.010922, valid_acc: 1.000000
epoch: 164, step: 460, time: 31.7, train: 0.010236, valid: 1.000000, train_acc: 0.010548, valid_acc: 1.000000
epoch: 165, step: 463, time: 44.0, train: 0.010120, valid: 1.000000, train_acc: 0.010181, valid_acc: 1.000000
epoch: 166, step: 466, time: 38.1, train: 0.009755, valid: 1.000000, train_acc: 0.009826, valid_acc: 1.000000
epoch: 167, step: 469, time: 40.8, train: 0.009350, valid: 1.000000, train_acc: 0.009485, valid_acc: 1.000000
epoch: 168, step: 472, time: 34.5, train: 0.009025, valid: 1.000000, train_acc: 0.009165, valid_acc: 1.000000
epoch: 169, step: 475, time: 35.7, train: 0.008166, valid: 1.000000, train_acc: 0.008859, valid_acc: 1.000000
epoch: 170, step: 478, time: 29.7, train: 0.007860, valid: 1.000000, train_acc: 0.008555, valid_acc: 1.000000
epoch: 171, step: 480, time: 31.3, train: 0.008148, valid: 1.000000, train_acc: 0.008353, valid_acc: 1.000000
epoch: 172, step: 482, time: 27.6, train: 0.007956, valid: 1.000000, train_acc: 0.008167, valid_acc: 1.000000
epoch: 173, step: 485, time: 45.8, train: 0.007329, valid: 1.000000, train_acc: 0.008225, valid_acc: 1.000000
epoch: 174, step: 488, time: 40.2, train: 0.007058, valid: 1.000000, train_acc: 0.007951, valid_acc: 1.000000
epoch: 175, step: 490, time: 34.0, train: 0.007190, valid: 1.000000, train_acc: 0.007457, valid_acc: 1.000000
epoch: 176, step: 492, time: 28.2, train: 0.007037, valid: 1.000000, train_acc: 0.007294, valid_acc: 1.000000
epoch: 177, step: 494, time: 26.2, train: 0.006544, valid: 1.000000, train_acc: 0.007441, valid_acc: 1.000000
epoch: 178, step: 496, time: 23.4, train: 0.006406, valid: 1.000000, train_acc: 0.007274, valid_acc: 1.000000
epoch: 179, step: 498, time: 25.1, train: 0.006211, valid: 1.000000, train_acc: 0.006821, valid_acc: 1.000000
epoch: 180, step: 500, time: 21.2, train: 0.006091, valid: 1.000000, train_acc: 0.006671, valid_acc: 1.000000
epoch: 181, step: 503, time: 38.1, train: 0.006336, valid: 1.000000, train_acc: 0.006444, valid_acc: 1.000000
epoch: 182, step: 506, time: 34.4, train: 0.006110, valid: 1.000000, train_acc: 0.006232, valid_acc: 1.000000
epoch: 183, step: 508, time: 28.5, train: 0.005857, valid: 1.000000, train_acc: 0.006090, valid_acc: 1.000000
epoch: 184, step: 510, time: 24.9, train: 0.005717, valid: 1.000000, train_acc: 0.005952, valid_acc: 1.000000
epoch: 185, step: 513, time: 38.3, train: 0.005615, valid: 1.000000, train_acc: 0.005753, valid_acc: 1.000000
epoch: 186, step: 516, time: 33.1, train: 0.005439, valid: 1.000000, train_acc: 0.005558, valid_acc: 1.000000
epoch: 187, step: 519, time: 37.2, train: 0.005225, valid: 1.000000, train_acc: 0.005373, valid_acc: 1.000000
epoch: 188, step: 522, time: 32.4, train: 0.005050, valid: 1.000000, train_acc: 0.005192, valid_acc: 1.000000
epoch: 189, step: 525, time: 37.9, train: 0.004525, valid: 1.000000, train_acc: 0.005257, valid_acc: 1.000000
epoch: 190, step: 528, time: 33.9, train: 0.004349, valid: 1.000000, train_acc: 0.005085, valid_acc: 1.000000
epoch: 191, step: 531, time: 40.7, train: 0.004644, valid: 1.000000, train_acc: 0.004688, valid_acc: 1.000000
epoch: 192, step: 534, time: 34.5, train: 0.004482, valid: 1.000000, train_acc: 0.004537, valid_acc: 1.000000
epoch: 193, step: 537, time: 34.8, train: 0.003932, valid: 1.000000, train_acc: 0.004377, valid_acc: 1.000000
epoch: 194, step: 540, time: 29.0, train: 0.003798, valid: 1.000000, train_acc: 0.004233, valid_acc: 1.000000
epoch: 195, step: 543, time: 37.5, train: 0.003834, valid: 1.000000, train_acc: 0.004303, valid_acc: 1.000000
epoch: 196, step: 546, time: 31.6, train: 0.003692, valid: 1.000000, train_acc: 0.004164, valid_acc: 1.000000
epoch: 197, step: 549, time: 40.7, train: 0.003580, valid: 1.000000, train_acc: 0.003996, valid_acc: 1.000000
epoch: 198, step: 552, time: 34.7, train: 0.003463, valid: 1.000000, train_acc: 0.003866, valid_acc: 1.000000
epoch: 199, step: 555, time: 34.6, train: 0.003258, valid: 1.000000, train_acc: 0.003768, valid_acc: 1.000000
epoch: 200, step: 558, time: 29.9, train: 0.003148, valid: 1.000000, train_acc: 0.003648, valid_acc: 1.000000
"""  # Include all the data here

# Parse the data
epochs = []
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

for line in data.strip().split('\n'):
    parts = line.split(', ')
    epoch = int(parts[0].split(': ')[1])
    train_loss = float(parts[3].split(': ')[1])
    valid_loss = float(parts[4].split(': ')[1])
    train_acc = float(parts[5].split(': ')[1])
    valid_acc = float(parts[6].split(': ')[1])
    
    epochs.append(epoch)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)

# Plot train vs. validation loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', color='blue')
plt.plot(epochs, valid_losses, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss (L-Chiral Sample Data)')
plt.legend()
plt.grid(True)

# Plot train vs. validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
plt.plot(epochs, valid_accuracies, label='Validation Accuracy', color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy (L-Chiral Sample Data)')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the plot to a file
plt.savefig('training_validation_plots_lchiral_sample_data.png')
print("Plot saved as 'training_validation_plots_lchiral_sample_data.png'")

