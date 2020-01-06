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

// Visualize creates a png of the matrix profile given a matrix profile.
func (mp MatrixProfile) Visualize(fn string, motifs []MotifGroup, discords []int, cac []float64) error {
	sigPts := points(mp.A, len(mp.A))
	mpPts := points(mp.MP, len(mp.A))
	cacPts := points(cac, len(mp.A))
	motifPts := make([][]plotter.XYs, len(motifs))
	discordPts := make([]plotter.XYs, len(discords))
	discordLabels := make([]string, len(discords))

	for i := 0; i < len(motifs); i++ {
		motifPts[i] = make([]plotter.XYs, len(motifs[i].Idx))
	}

	for i := 0; i < len(motifs); i++ {
		for j, idx := range motifs[i].Idx {
			motifPts[i][j] = points(mp.A[idx:idx+mp.M], mp.M)
		}
	}

	for i, idx := range discords {
		discordPts[i] = points(mp.A[idx:idx+mp.M], mp.M)
		discordLabels[i] = strconv.Itoa(idx)
	}

	return plotMP(sigPts, mpPts, cacPts, motifPts, discordPts, discordLabels, fn)
}

func points(a []float64, n int) plotter.XYs {
	pts := make(plotter.XYs, n)
	for i := 0; i < n; i++ {
		pts[i].X = float64(i)
		if i < len(a) {
			pts[i].Y = a[i]
		}
	}
	return pts
}

func createPlot(pts []plotter.XYs, labels []string, title string) (*plot.Plot, error) {
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

func plotMP(sigPts, mpPts, cacPts plotter.XYs, motifPts [][]plotter.XYs, discordPts []plotter.XYs, discordLabels []string, filename string) error {
	var err error
	rows, cols := len(motifPts), 2
	if rows < 4 {
		rows = 4
	}
	plots := make([][]*plot.Plot, rows)

	for i := 0; i < rows; i++ {
		plots[i] = make([]*plot.Plot, cols)
	}

	plots[0][0], err = createPlot([]plotter.XYs{sigPts}, nil, "signal")
	if err != nil {
		return err
	}

	plots[1][0], err = createPlot([]plotter.XYs{mpPts}, nil, "matrix profile")
	if err != nil {
		return err
	}

	plots[2][0], err = createPlot([]plotter.XYs{cacPts}, nil, "corrected arc curve")
	if err != nil {
		return err
	}

	plots[3][0], err = createPlot(discordPts, discordLabels, "discords")
	if err != nil {
		return err
	}

	for i := 0; i < len(motifPts); i++ {
		plots[i][1], err = createPlot(motifPts[i], nil, fmt.Sprintf("motif %d", i))
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

// Visualize creates a png of the k-dimensional matrix profile.
func (mp KMatrixProfile) Visualize(fn string) error {
	sigPts := make([]plotter.XYs, len(mp.T))
	for i := 0; i < len(mp.T); i++ {
		sigPts[i] = points(mp.T[i], len(mp.T[0]))
	}

	mpPts := make([]plotter.XYs, len(mp.MP))
	for i := 0; i < len(mp.MP); i++ {
		mpPts[i] = points(mp.MP[i], len(mp.T[0]))
	}

	return mp.plotMP(sigPts, mpPts, fn)
}

func (mp KMatrixProfile) plotMP(sigPts, mpPts []plotter.XYs, filename string) error {
	var err error

	rows, cols := len(sigPts)*2, 1

	plots := make([][]*plot.Plot, rows)

	for i := 0; i < len(sigPts)*2; i++ {
		plots[i] = make([]*plot.Plot, cols)
	}

	for i := 0; i < len(sigPts); i++ {
		plots[i][0], err = createPlot([]plotter.XYs{sigPts[i]}, nil, fmt.Sprintf("signal%d", i))
		if err != nil {
			return err
		}
	}

	for i := 0; i < len(sigPts); i++ {
		plots[len(sigPts)+i][0], err = createPlot([]plotter.XYs{mpPts[i]}, nil, fmt.Sprintf("mp%d", i))
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
