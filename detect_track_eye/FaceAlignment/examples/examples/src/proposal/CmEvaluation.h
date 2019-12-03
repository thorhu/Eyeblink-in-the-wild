#pragma once

struct CmEvaluation
{
	// Save the precision recall curve, and ROC curve to a Matlab file: resName
	// Return area under ROC curve
	static void Evaluate(CStr gtW, CStr &salDir, CStr &resName, vecS &des); 
	static void Evaluate(CStr gtW, CStr &salDir, CStr &resName, CStr &des) {vecS descri(1); descri[0] = des; Evaluate(gtW, salDir, resName, descri);} 

	static void EvalueMask(CStr gtW, CStr &maskDir, CStr &gtExt, CStr &maskExt, bool back = false, bool alertNul = true);

	static void MeanAbsoluteError(CStr &inDir, CStr &salDir, vecS &des);

	static int STEP; // Evaluation threshold density
private:
	static const int COLOR_NUM = 255;  
	static const int MI;  // Number of difference threshold

	static void PrintVector(FILE *f, const vecD &v, CStr &name);

	static int Evaluate_(CStr &gtImgW, CStr &inDir, CStr& resExt, vecD &precision, vecD &recall, vecD &tpr, vecD &fpr);
};

