package matrixprofile

// AnalyzeOpts contains all the parameters needed for basic features to discover from
// a matrix profile. This is currently limited to motif, discord, and segmentation discovery.
type AnalyzeOpts struct {
	kMotifs        int     // the top k motifs to find
	rMotifs        float64 // the max radius to find motifs
	kDiscords      int     // the top k discords to find
	OutputFilename string  // relative or absolute filepath for the visualization output
}

// NewAnalyzeOpts creates a default set of parameters to analyze the matrix profile.
func NewAnalyzeOpts() *AnalyzeOpts {
	return &AnalyzeOpts{
		kMotifs:        3,
		rMotifs:        2,
		kDiscords:      3,
		OutputFilename: "mp.png",
	}
}
