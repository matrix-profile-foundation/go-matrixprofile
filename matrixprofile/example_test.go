package matrixprofile

/*
import (
	"bufio"
	"encoding/csv"
	"io"
	"os"
	"strconv"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

func readPoints(numRecords int) (plotter.XYs, plotter.XYs) {
	var val float64

	csvFile, _ := os.Open("testdata/id-1003_heartrate_seconds_20171001_20171007.csv")
	r := csv.NewReader(bufio.NewReader(csvFile))

	var records []float64
	var numRec int
	for {
		if numRec >= numRecords {
			break
		}
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			panic(err)
		}

		val, err = strconv.ParseFloat(record[1], 64)
		if err != nil {
			continue
		}
		records = append(records, val)
		numRec++
	}

	rawPts := make(plotter.XYs, len(records))
	for i, val := range records {
		rawPts[i].X = float64(i)
		rawPts[i].Y = val
	}

	mp, mpIdx, err := Stmp(records, nil, 20)
	if err != nil {
		panic(err)
	}

	mpPts := make(plotter.XYs, len(records))
	for i := range records {
		mpPts[i].X = float64(i)
		if i < len(mp) {
			mpPts[i].Y = mp[i]
		}
	}

	return rawPts, mpPts
}

func Example() {
	raw, mp := readPoints(500)

	rows, cols := 2, 1
	plots := make([][]*plot.Plot, rows)
	plots[0] = make([]*plot.Plot, cols)
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	err = plotutil.AddLines(p,
		"data", raw,
	)
	if err != nil {
		panic(err)
	}

	plots[0][0] = p

	plots[1] = make([]*plot.Plot, cols)
	p, err = plot.New()
	if err != nil {
		panic(err)
	}
	err = plotutil.AddLines(p,
		"matrix profile", mp,
	)
	if err != nil {
		panic(err)
	}

	plots[1][0] = p

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

	w, err := os.Create("mp.png")
	if err != nil {
		panic(err)
	}

	png := vgimg.PngCanvas{Canvas: img}
	if _, err := png.WriteTo(w); err != nil {
		panic(err)
	}
	// Output: something
}
*/
