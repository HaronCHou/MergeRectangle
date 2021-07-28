#pragma once
#include<opencv2/opencv.hpp>
#include "labels_solver.h"

class SAUF {
public:
	cv::Mat1b img_;
	cv::Mat1i img_labels_;
	unsigned int n_labels_;

	UFPC ufpc_solver;
	SAUF(cv::Mat1b img)
	{
		img_ = img;
	}

#define UPPER_BOUND_8_CONNECTIVITY ((size_t)((img_.rows + 1) / 2) * (size_t)((img_.cols + 1) / 2) + 1)

	void PerformLabeling()
	{
		const int h = img_.rows;
		const int w = img_.cols;

		img_labels_ = cv::Mat1i(img_.size(), 0); // Allocation + initialization of the output image

		ufpc_solver.Alloc(UPPER_BOUND_8_CONNECTIVITY); // Memory allocation of the labels solver
		ufpc_solver.Setup(); // Labels solver initialization
							 // First scan
		for (int r = 0; r < h; ++r) {
			// Get row pointers
			unsigned char const * const img_row = img_.ptr<unsigned char>(r);
			unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
			unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(r);
			unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);

			for (int c = 0; c < w; ++c) {
#define CONDITION_P c > 0 && r > 0 && img_row_prev[c - 1] > 0
#define CONDITION_Q r > 0 && img_row_prev[c] > 0
#define CONDITION_R c < w - 1 && r > 0 && img_row_prev[c + 1] > 0
#define CONDITION_S c > 0 && img_row[c - 1] > 0
#define CONDITION_X img_row[c] > 0

#define ACTION_1 // nothing to do 
#define ACTION_2 img_labels_row[c] = ufpc_solver.NewLabel(); // new label
#define ACTION_3 img_labels_row[c] = img_labels_row_prev[c - 1]; // x <- p
#define ACTION_4 img_labels_row[c] = img_labels_row_prev[c]; // x <- q
#define ACTION_5 img_labels_row[c] = img_labels_row_prev[c + 1]; // x <- r
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 1]; // x <- s
#define ACTION_7 img_labels_row[c] = ufpc_solver.Merge(img_labels_row_prev[c - 1], img_labels_row_prev[c + 1]); // x <- p + r
#define ACTION_8 img_labels_row[c] = ufpc_solver.Merge(img_labels_row[c - 1], img_labels_row_prev[c + 1]); // x <- s + r

#include "labeling_wu_2009_tree.inc.h"
			}
		}

		// Second scan
		n_labels_ = ufpc_solver.Flatten();

		for (int r = 0; r < img_labels_.rows; ++r) {
			unsigned * img_row_start = img_labels_.ptr<unsigned>(r);
			unsigned * const img_row_end = img_row_start + img_labels_.cols;
			for (; img_row_start != img_row_end; ++img_row_start) {
				*img_row_start = ufpc_solver.GetLabel(*img_row_start);
			}
		}
#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8

#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_X

	}

};
