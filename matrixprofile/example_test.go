package matrixprofile

import (
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

func createPoints(sig []float64) (plotter.XYs, plotter.XYs, plotter.XYs, error) {
	rawPts := make(plotter.XYs, len(sig))
	for i, val := range sig {
		rawPts[i].X = float64(i)
		rawPts[i].Y = val
	}

	mp, err := New(sig, nil, 32)
	if err != nil {
		return nil, nil, nil, err
	}

	if err = mp.Stmp(); err != nil {
		return nil, nil, nil, err
	}

	_, _, cac := mp.Segment()

	mpPts := make(plotter.XYs, len(sig))
	for i := range sig {
		mpPts[i].X = float64(i)
		if i < len(mp.MP) {
			mpPts[i].Y = mp.MP[i]
		}
	}

	cacPts := make(plotter.XYs, len(sig))
	for i := range sig {
		cacPts[i].X = float64(i)
		if i < len(cac) {
			cacPts[i].Y = cac[i]
		}
	}

	return rawPts, mpPts, cacPts, err
}

func plotMP(raw, mp, cac plotter.XYs, filename string) error {
	rows, cols := 3, 1
	plots := make([][]*plot.Plot, rows)

	plots[0] = make([]*plot.Plot, cols)
	p, err := plot.New()
	if err != nil {
		return err
	}
	err = plotutil.AddLines(p,
		"data", raw,
	)
	if err != nil {
		return err
	}

	plots[0][0] = p

	plots[1] = make([]*plot.Plot, cols)
	p, err = plot.New()
	if err != nil {
		return err
	}
	err = plotutil.AddLines(p,
		"matrix profile", mp,
	)
	if err != nil {
		return err
	}

	plots[1][0] = p

	plots[2] = make([]*plot.Plot, cols)
	p, err = plot.New()
	if err != nil {
		return err
	}
	err = plotutil.AddLines(p,
		"cac", cac,
	)
	if err != nil {
		return err
	}

	plots[2][0] = p

	img := vgimg.New(vg.Points(1200), vg.Points(1200))
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
	sin := generateSin(1, 5, 0, 0, 100, 2)
	sin2 := generateSin(0.25, 10, 0, 0.75, 100, 1)
	sig := append(sin, sin2...)
	noise := generateNoise(0.1, len(sig))
	sig = sigAdd(sig, noise)

	raw, mp, cac, err := createPoints(sig)
	if err != nil {
		panic(err)
	}
	if err = plotMP(raw, mp, cac, "mp_sine.png"); err != nil {
		panic(err)
	}

	saw := generateSawtooth(1, 5, 0, 0, 100, 2)
	ext := generateLine(0.08, -1, len(saw)/2)
	sig = append(saw, ext...)
	noise = generateNoise(0.1, len(sig))
	sig = sigAdd(sig, noise)

	raw, mp, cac, err = createPoints(sig)
	if err != nil {
		panic(err)
	}
	if err = plotMP(raw, mp, cac, "mp_sawtooth.png"); err != nil {
		panic(err)
	}

	line := generateLine(0, 0, 120)
	ext = generateLine(0, 100, len(line)/2)
	ext2 := generateLine(0, 600, len(line)/2)
	sig = append(line, ext...)
	sig = append(sig, ext2...)
	noise = generateNoise(10, len(sig))
	sig = sigAdd(sig, noise)

	raw, mp, cac, err = createPoints(sig)
	if err != nil {
		panic(err)
	}
	if err = plotMP(raw, mp, cac, "mp_rect.png"); err != nil {
		panic(err)
	}

	// Output:
}
