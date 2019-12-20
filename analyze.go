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
func NewAnalyzeOpts() AnalyzeOptions {
	return AnalyzeOptions{
		KMotifs:        3,
		RMotifs:        2,
		KDiscords:      3,
		OutputFilename: "mp.png",
	}
}

// Analyze performs the matrix profile computation and discovers various features
// from the profile such as motifs, discords, and segmentation. The results are
// visualized and saved into an output file.
func (mp MatrixProfile) Analyze(co ComputeOptions, ao AnalyzeOptions) error {
	var err error
	if err = mp.Compute(co); err != nil {
		return err
	}

	_, _, cac := mp.DiscoverSegments()

	motifs, err := mp.DiscoverMotifs(ao.KMotifs, ao.RMotifs)
	if err != nil {
		return err
	}

	discords, err := mp.DiscoverDiscords(ao.KDiscords, mp.M/2)
	if err != nil {
		return err
	}

	return mp.Visualize(ao.OutputFilename, motifs, discords, cac)
}
