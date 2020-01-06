package matrixprofile

import (
	"fmt"

	"github.com/matrix-profile-foundation/go-matrixprofile/siggen"
)

/*
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

func PlotKMP(sigPts, mpPts []plotter.XYs, filename string) error {
	var err error
	rows, cols := len(sigPts)*2, 1

	plots := make([][]*plot.Plot, rows)

	for i := 0; i < len(sigPts)*2; i++ {
		plots[i] = make([]*plot.Plot, cols)
	}

	for i := 0; i < len(sigPts); i++ {
		plots[i][0], err = CreatePlot([]plotter.XYs{sigPts[i]}, nil, fmt.Sprintf("signal%d", i))
		if err != nil {
			return err
		}
	}

	for i := 0; i < len(sigPts); i++ {
		plots[len(sigPts)+i][0], err = CreatePlot([]plotter.XYs{mpPts[i]}, nil, fmt.Sprintf("mp%d", i))
		if err != nil {
			return err
		}
	}

	img := vgimg.New(vg.Points(600), vg.Points(600))
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
*/

func Example_caseStudy() {
	sin := siggen.Sin(1, 5, 0, 0, 100, 2)
	sin2 := siggen.Sin(0.25, 10, 0, 0.75, 100, 0.25)
	saw := siggen.Sawtooth(0.5, 7, 0, 0, 100, 1)
	line := siggen.Line(0, 0, len(sin2)*4)
	sig := siggen.Append(sin, sin2, sin, line, sin2, line, sin2, line, saw)

	noise := siggen.Noise(0.1, len(sig))
	sig = siggen.Add(sig, noise)

	var m, k int
	var r float64
	m = 32
	k = 6
	r = 3
	mp, err := New(sig, nil, m)
	if err != nil {
		panic(err)
	}

	ao := NewAnalyzeOpts()
	ao.KMotifs = k
	ao.RMotifs = r
	ao.OutputFilename = "mp_sine.png"

	if err = mp.Analyze(nil, ao); err != nil {
		panic(err)
	}

	fmt.Printf("Saved png file result to %s\n", ao.OutputFilename)
	// Output: Saved png file result to mp_sine.png
}

func Example_kDimensionalCaseStudy() {
	sin := siggen.Sin(1, 4, 0, 0, 100, 0.25)
	saw := siggen.Sawtooth(1, 4, 0, 0, 100, 0.25)
	square := siggen.Square(1, 4, 0, 0, 100, 0.25)
	line := siggen.Line(0, 0, len(sin)*4)
	line2 := siggen.Line(0, 0, len(sin)*3)
	sig := make([][]float64, 3)
	sig[0] = siggen.Append(line, line, line, saw, line2, saw, line2)
	sig[1] = siggen.Append(line, sin, line2, sin, line2, sin, line2, sin, line2)
	sig[2] = siggen.Append(line, square, line2, square, line2, square, line2, square, line2)

	sig[0] = siggen.Add(sig[0], siggen.Noise(0.1, len(sig[0])))
	sig[1] = siggen.Add(sig[1], siggen.Noise(0.1, len(sig[0])))
	sig[2] = siggen.Add(sig[2], siggen.Noise(0.1, len(sig[0])))

	m := 25
	mp, err := NewK(sig, m)
	if err != nil {
		panic(err)
	}

	if err = mp.Compute(); err != nil {
		panic(err)
	}

	if err = mp.Visualize("mp_kdim.png"); err != nil {
		panic(err)
	}

	fmt.Println("Saved png file result to mp_kdim.png")
	// Output: Saved png file result to mp_kdim.png
}
