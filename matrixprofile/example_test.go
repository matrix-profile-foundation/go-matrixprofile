package matrixprofile

import (
	"fmt"
	"os"
	"strconv"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

func Points(a []float64, n int) plotter.XYs {
	pts := make(plotter.XYs, n)
	for i := 0; i < n; i++ {
		pts[i].X = float64(i)
		if i < len(a) {
			pts[i].Y = a[i]
		}
	}
	return pts
}

func CreatePlot(pts []plotter.XYs, labels []string, title string) (*plot.Plot, error) {
	if labels != nil && len(pts) != len(labels) {
		return nil, fmt.Errorf("number of XYs, %d, does not match number of labels, %d", len(pts), len(labels))
	}

	p, err := plot.New()
	if err != nil {
		return p, err
	}

	p.Title.Text = title
	for i := 0; i < len(pts); i++ {
		line, points, err := plotter.NewLinePoints(pts[i])
		if err != nil {
			return p, err
		}
		line.Color = plotutil.Color(i)
		points.Color = plotutil.Color(i)
		points.Shape = nil
		p.Add(line, points)
		if labels != nil {
			p.Legend.Add(labels[i], line)
		}
	}
	return p, err
}

func PlotMP(sigPts, mpPts, cacPts plotter.XYs, motifPts [][]plotter.XYs, discordPts []plotter.XYs, discordLabels []string, filename string) error {
	var err error
	rows, cols := len(motifPts), 2
	if rows < 4 {
		rows = 4
	}
	plots := make([][]*plot.Plot, rows)

	for i := 0; i < len(motifPts); i++ {
		plots[i] = make([]*plot.Plot, cols)
	}

	plots[0][0], err = CreatePlot([]plotter.XYs{sigPts}, nil, "signal")
	if err != nil {
		return err
	}

	plots[1][0], err = CreatePlot([]plotter.XYs{mpPts}, nil, "matrix profile")
	if err != nil {
		return err
	}

	plots[2][0], err = CreatePlot([]plotter.XYs{cacPts}, nil, "corrected arc curve")
	if err != nil {
		return err
	}

	plots[3][0], err = CreatePlot(discordPts, discordLabels, "discords")
	if err != nil {
		return err
	}

	for i := 0; i < len(motifPts); i++ {
		plots[i][1], err = CreatePlot(motifPts[i], nil, fmt.Sprintf("motif %d", i))
		if err != nil {
			return err
		}
	}

	img := vgimg.New(vg.Points(1200), vg.Points(600))
	dc := draw.New(img)

	t := draw.Tiles{
		Rows: rows,
		Cols: cols,
	}

	canvases := plot.Align(plots, t, dc)
	for j := 0; j < rows; j++ {
		for i := 0; i < cols; i++ {
			if plots[j][i] != nil {
				plots[j][i].Draw(canvases[j][i])
			}
		}
	}

	w, err := os.Create(filename)
	if err != nil {
		return err
	}

	png := vgimg.PngCanvas{Canvas: img}
	_, err = png.WriteTo(w)
	return err
}

func Example() {
	sin := Sin(1, 5, 0, 0, 100, 2)
	sin2 := Sin(0.25, 10, 0, 0.75, 100, 0.25)
	saw := Sawtooth(0.5, 7, 0, 0, 100, 1)
	noise := Noise(0.3, len(sin2)*4)

	sig := append(sin, sin2...)
	sig = append(sig, sin...)
	sig = append(sig, noise...)
	sig = append(sig, sin2...)
	sig = append(sig, noise...)
	sig = append(sig, sin2...)
	sig = append(sig, noise...)
	sig = append(sig, saw...)

	noise = Noise(0.1, len(sig))
	sig = SigAdd(sig, noise)

	var m, k int
	var r float64
	m = 32
	k = 6
	r = 3
	mp, err := New(sig, nil, m)
	if err != nil {
		panic(err)
	}

	if err = mp.Stomp(2); err != nil {
		panic(err)
	}

	_, _, cac := mp.Segment()

	motifs, err := mp.TopKMotifs(k, r)
	if err != nil {
		panic(err)
	}

	discords := mp.Discords(3, mp.m/2)
	if err != nil {
		panic(err)
	}

	sigPts := Points(sig, len(sig))
	mpPts := Points(mp.MP, len(sig))
	cacPts := Points(cac, len(sig))
	motifPts := make([][]plotter.XYs, k)
	discordPts := make([]plotter.XYs, k)
	discordLabels := make([]string, k)

	for i := 0; i < k; i++ {
		motifPts[i] = make([]plotter.XYs, len(motifs[i].Idx))
	}

	for i := 0; i < k; i++ {
		for j, idx := range motifs[i].Idx {
			motifPts[i][j] = Points(sig[idx:idx+m], m)
		}
	}

	for i, idx := range discords {
		discordPts[i] = Points(sig[idx:idx+m], m)
		discordLabels[i] = strconv.Itoa(idx)
	}

	if err = PlotMP(sigPts, mpPts, cacPts, motifPts, discordPts, discordLabels, "../mp_sine.png"); err != nil {
		panic(err)
	}

	fmt.Println("Saved png file result to mp_sine.png")
	// Output: Saved png file result to mp_sine.png
}

func ExampleMatrixProfile_Stmp() {
	// generate a signal mainly composed of sine waves and switches
	// frequencies, amplitude, and offset midway through

	// amplitude of 1, frequency of 5Hz, sampling frequency of 100 Hz,
	// time of 2 seconds
	sin := Sin(1, 5, 0, 0, 100, 2)

	// amplitude of 0.25, frequency of 10Hz, offset of 0.75, sampling
	// frequency of 100 Hz, time of 1 second
	sin2 := Sin(0.25, 10, 0, 0.75, 100, 1)
	sig := append(sin, sin2...)

	// noise with an amplitude of 0.1
	noise := Noise(0.1, len(sig))
	sig = SigAdd(sig, noise)

	// create a new MatrixProfile struct using the signal and a
	// subsequence length of 32. The second subsequence is set to nil
	// so we perform a self join.
	mp, err := New(sig, nil, 32)
	if err != nil {
		panic(err)
	}

	// run the STMP algorithm with self join. The matrix profile
	// will be stored in mp.MP and the matrix profile index will
	// be stored in mp.Idx
	if err = mp.Stmp(); err != nil {
		panic(err)
	}
}

func ExampleMatrixProfile_Stamp() {
	// generate a signal mainly composed of sine waves and switches
	// frequencies, amplitude, and offset midway through

	// amplitude of 1, frequency of 5Hz, sampling frequency of 100 Hz,
	// time of 2 seconds
	sin := Sin(1, 5, 0, 0, 100, 2)

	// amplitude of 0.25, frequency of 10Hz, offset of 0.75, sampling
	// frequency of 100 Hz, time of 1 second
	sin2 := Sin(0.25, 10, 0, 0.75, 100, 1)
	sig := append(sin, sin2...)

	// noise with an amplitude of 0.1
	noise := Noise(0.1, len(sig))
	sig = SigAdd(sig, noise)

	// create a new MatrixProfile struct using the signal and a
	// subsequence length of 32. The second subsequence is set to nil
	// so we perform a self join.
	mp, err := New(sig, nil, 32)
	if err != nil {
		panic(err)
	}

	// run the STAMP algorithm with self join and a sample of 0.2 of
	// all subsequences. The matrix profile will be stored in mp.MP
	// and the matrix profile index will be stored in mp.Idx
	if err = mp.Stamp(0.2); err != nil {
		panic(err)
	}

}

func ExampleMatrixProfile_Stomp() {
	// generate a signal mainly composed of sine waves and switches
	// frequencies, amplitude, and offset midway through

	// amplitude of 1, frequency of 5Hz, sampling frequency of 100 Hz,
	// time of 2 seconds
	sin := Sin(1, 5, 0, 0, 100, 2)

	// amplitude of 0.25, frequency of 10Hz, offset of 0.75, sampling
	// frequency of 100 Hz, time of 1 second
	sin2 := Sin(0.25, 10, 0, 0.75, 100, 1)
	sig := append(sin, sin2...)

	// noise with an amplitude of 0.1
	noise := Noise(0.1, len(sig))
	sig = SigAdd(sig, noise)

	// create a new MatrixProfile struct using the signal and a
	// subsequence length of 32. The second subsequence is set to nil
	// so we perform a self join.
	mp, err := New(sig, nil, 32)
	if err != nil {
		panic(err)
	}

	// run the STOMP algorithm with self join. The matrix profile
	// will be stored in mp.MP and the matrix profile index will
	// be stored in mp.Idx
	if err = mp.Stomp(1); err != nil {
		panic(err)
	}
}

func ExampleMatrixProfile_Segment() {
	// generate a signal mainly composed of sine waves and switches
	// frequencies, amplitude, and offset midway through

	// amplitude of 1, frequency of 5Hz, sampling frequency of 100 Hz,
	// time of 2 seconds
	sin := Sin(1, 5, 0, 0, 100, 2)

	// amplitude of 0.25, frequency of 10Hz, offset of 0.75, sampling
	// frequency of 100 Hz, time of 1 second
	sin2 := Sin(0.25, 10, 0, 0.75, 100, 1)
	sig := append(sin, sin2...)

	// noise with an amplitude of 0.1
	noise := Noise(0.01, len(sig))
	sig = SigAdd(sig, noise)

	// create a new MatrixProfile struct using the signal and a
	// subsequence length of 32. The second subsequence is set to nil
	// so we perform a self join.
	mp, err := New(sig, nil, 32)
	if err != nil {
		panic(err)
	}

	// run the STMP algorithm with self join. The matrix profile
	// will be stored in mp.MP and the matrix profile index will
	// be stored in mp.Idx
	if err = mp.Stmp(); err != nil {
		panic(err)
	}

	// segment the timeseries using the number of arc crossings over
	// each index in the matrix profile index
	idx, cac, _ := mp.Segment()
	fmt.Printf("Signal change foud at index: %d\n", idx)
	fmt.Printf("Corrected Arc Curve (CAC) value: %.3f\n", cac)

	// Output:
	// Signal change foud at index: 194
	// Corrected Arc Curve (CAC) value: 0.000
}

func ExampleMatrixProfile_TopKMotifs() {
	// generate a signal mainly composed of sine waves and switches
	// frequencies, amplitude, and offset midway through

	// amplitude of 1, frequency of 5Hz, sampling frequency of 100 Hz,
	// time of 2 seconds
	sin := Sin(1, 5, 0, 0, 100, 2)

	// amplitude of 0.25, frequency of 10Hz, offset of 0.75, sampling
	// frequency of 100 Hz, time of 1 second
	sin2 := Sin(0.25, 10, 0, 0.75, 100, 1)
	sig := append(sin, sin2...)

	// create a new MatrixProfile struct using the signal and a
	// subsequence length of 32. The second subsequence is set to nil
	// so we perform a self join.
	mp, err := New(sig, nil, 32)
	if err != nil {
		panic(err)
	}

	// run the STMP algorithm with self join. The matrix profile
	// will be stored in mp.MP and the matrix profile index will
	// be stored in mp.Idx
	if err = mp.Stmp(); err != nil {
		panic(err)
	}

	// finds the top 3 motifs in the signal. Motif groups include
	// all subsequences that are within 2 times the distance of the
	// original motif pair
	motifs, err := mp.TopKMotifs(2, 2)
	if err != nil {
		panic(err)
	}

	for i, mg := range motifs {
		fmt.Printf("Motif Group %d\n", i)
		fmt.Printf("  %d motifs\n", len(mg.Idx))
	}

	// Output:
	// Motif Group 0
	//   2 motifs
	// Motif Group 1
	//   2 motifs
}
