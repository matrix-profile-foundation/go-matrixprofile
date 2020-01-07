package matrixprofile

// AnalyzeOptions contains all the parameters needed for basic features to discover from
// a matrix profile. This is currently limited to motif, discord, and segmentation discovery.
type AnalyzeOptions struct {
	KMotifs        int     // the top k motifs to find
	RMotifs        float64 // the max radius to find motifs
	KDiscords      int     // the top k discords to find
	OutputFilename string  // relative or absolute filepath for the visualization output
}

// NewAnalyzeOpts creates a default set of parameters to analyze the matrix profile.
func NewAnalyzeOpts() *AnalyzeOptions {
	return &AnalyzeOptions{
		KMotifs:        3,
		RMotifs:        2,
		KDiscords:      3,
		OutputFilename: "mp.png",
	}
}
